//! Uniform blocks written by field name. Mirrors luma.gl's `UniformBlock` / `ShaderInputs`.
//! Float fields can animate: with a [`UniformTransition`] registered for a field, writes set a
//! target and the block writes the value on its way there at the block's time.

use std::collections::HashMap;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::shader::{UniformKind, UniformLayout};
use crate::{LumaError, Result};

/// How a uniform field moves to a new value.
#[derive(Clone, Copy, Debug)]
pub enum UniformTransition {
    /// Over `duration` seconds along an easing from 0 to 1
    Interpolation { duration: f64, easing: fn(f64) -> f64 },
    /// A spring stepped once per frame: `velocity = velocity * damping + (target - value) *
    /// stiffness`, then `value += velocity`
    Spring { stiffness: f64, damping: f64 },
}

impl PartialEq for UniformTransition {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Interpolation { duration, easing },
                Self::Interpolation {
                    duration: d,
                    easing: e,
                },
            ) => duration == d && std::ptr::fn_addr_eq(*easing, *e),
            (
                Self::Spring { stiffness, damping },
                Self::Spring {
                    stiffness: s,
                    damping: d,
                },
            ) => stiffness == s && damping == d,
            _ => false,
        }
    }
}

/// The animation of one float field.
#[derive(Clone, Debug)]
pub(crate) struct Animated {
    transition: UniformTransition,
    from: Vec<f32>,
    target: Vec<f32>,
    current: Vec<f32>,
    velocity: Vec<f32>,
    start: f64,
    last_step: f64,
    started: bool,
    active: bool,
}

impl Animated {
    pub(crate) fn new(transition: UniformTransition) -> Self {
        Self {
            transition,
            from: Vec::new(),
            target: Vec::new(),
            current: Vec::new(),
            velocity: Vec::new(),
            start: 0.0,
            last_step: f64::NAN,
            started: false,
            active: false,
        }
    }

    /// Take the value written for the field and return the one to store at `time`.
    pub(crate) fn advance(&mut self, target: &[f32], time: f64) -> Vec<f32> {
        if !self.started || target.len() != self.target.len() {
            // the first value is taken as is
            self.started = true;
            self.target = target.to_vec();
            self.current = target.to_vec();
            self.velocity = vec![0.0; target.len()];
            self.active = false;
            return self.current.clone();
        }
        if target != self.target.as_slice() {
            self.from = self.current.clone();
            self.target = target.to_vec();
            self.start = time;
            self.active = true;
        }
        if !self.active {
            return self.current.clone();
        }
        match self.transition {
            UniformTransition::Interpolation { duration, easing } => {
                let t = if duration > 0.0 {
                    ((time - self.start) / duration).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let eased = easing(t).clamp(0.0, 1.0) as f32;
                for ((current, from), to) in self.current.iter_mut().zip(&self.from).zip(&self.target) {
                    *current = from + (to - from) * eased;
                }
                if t >= 1.0 {
                    self.current.clone_from(&self.target);
                    self.active = false;
                }
            }
            UniformTransition::Spring { stiffness, damping } => {
                if time != self.last_step {
                    self.last_step = time;
                    let mut settled = true;
                    for ((current, velocity), to) in
                        self.current.iter_mut().zip(&mut self.velocity).zip(&self.target)
                    {
                        *velocity = *velocity * damping as f32 + (to - *current) * stiffness as f32;
                        *current += *velocity;
                        if (to - *current).abs() > 1e-5 || velocity.abs() > 1e-5 {
                            settled = false;
                        }
                    }
                    if settled {
                        self.current.clone_from(&self.target);
                        self.velocity.iter_mut().for_each(|v| *v = 0.0);
                        self.active = false;
                    }
                }
            }
        }
        self.current.clone()
    }

    pub(crate) fn active(&self) -> bool {
        self.active
    }
}

/// CPU shadow of a WGSL uniform struct plus the GPU buffer it is uploaded to.
#[derive(Debug)]
pub struct UniformBlock {
    layout: UniformLayout,
    data: Vec<u8>,
    buffer: wgpu::Buffer,
    dirty: bool,
    transitions: HashMap<String, Animated>,
    time: f64,
}

fn round_up(n: u32, align: u32) -> u32 {
    n.div_ceil(align) * align
}

impl UniformBlock {
    pub fn new(device: &wgpu::Device, label: &str, layout: &UniformLayout) -> Self {
        let size = round_up(layout.size.max(16), 16);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            layout: layout.clone(),
            data: vec![0; size as usize],
            buffer,
            // Upload zeros on first use so the buffer is always initialized.
            dirty: true,
            transitions: HashMap::new(),
            time: 0.0,
        }
    }

    /// Animate writes to a float field (`f32` to `vec4<f32>`) with `transition`, or stop
    /// animating it with `None`. A running animation keeps going when the same transition is
    /// registered again.
    pub fn set_transition(&mut self, name: &str, transition: Option<UniformTransition>) {
        match transition {
            Some(transition) => match self.transitions.get_mut(name) {
                Some(animated) if animated.transition == transition => {}
                Some(animated) => animated.transition = transition,
                None => {
                    self.transitions
                        .insert(name.to_string(), Animated::new(transition));
                }
            },
            None => {
                self.transitions.remove(name);
            }
        }
    }

    /// The time animated fields are evaluated at, in seconds; set it before the writes of a frame.
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Whether a field is still moving towards its value.
    pub fn in_transition(&self) -> bool {
        self.transitions.values().any(Animated::active)
    }

    /// Whether writes to `name` are animated.
    pub fn has_transition(&self, name: &str) -> bool {
        self.transitions.contains_key(name)
    }

    pub fn layout(&self) -> &UniformLayout {
        &self.layout
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// The CPU copy of the block, laid out as the shader expects.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Write pending changes to the GPU. Must be called before the buffer is used in a pass.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty {
            queue.write_buffer(&self.buffer, 0, &self.data);
            self.dirty = false;
        }
    }

    fn write(&mut self, name: &str, bytes: &[u8], kinds: &[UniformKind]) -> Result<()> {
        let field = self.layout.field(name).ok_or_else(|| {
            LumaError::Uniform(format!(
                "no field `{name}` in uniform struct `{}`",
                self.layout.struct_name
            ))
        })?;
        if !kinds.is_empty() && !kinds.contains(&field.kind) {
            return Err(LumaError::Uniform(format!(
                "field `{name}` in `{}` has kind {:?}, cannot write {:?}",
                self.layout.struct_name, field.kind, kinds
            )));
        }
        if bytes.len() > field.size as usize {
            return Err(LumaError::Uniform(format!(
                "field `{name}` in `{}` is {} bytes, cannot write {} bytes",
                self.layout.struct_name,
                field.size,
                bytes.len()
            )));
        }
        let start = field.offset as usize;
        let animated = matches!(
            field.kind,
            UniformKind::F32 | UniformKind::Vec2F | UniformKind::Vec3F | UniformKind::Vec4F
        ) && bytes.len().is_multiple_of(4);
        if animated {
            if let Some(animation) = self.transitions.get_mut(name) {
                let target: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let value = animation.advance(&target, self.time);
                for (i, v) in value.iter().enumerate() {
                    self.data[start + i * 4..start + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                self.dirty = true;
                return Ok(());
            }
        }
        self.data[start..start + bytes.len()].copy_from_slice(bytes);
        self.dirty = true;
        Ok(())
    }

    pub fn set_f32(&mut self, name: &str, value: f32) -> Result<()> {
        self.write(name, &value.to_le_bytes(), &[UniformKind::F32])
    }

    pub fn set_i32(&mut self, name: &str, value: i32) -> Result<()> {
        self.write(name, &value.to_le_bytes(), &[UniformKind::I32])
    }

    pub fn set_u32(&mut self, name: &str, value: u32) -> Result<()> {
        self.write(name, &value.to_le_bytes(), &[UniformKind::U32])
    }

    /// Write a boolean into an `f32`, `i32`, `u32` or `bool` field as 1 or 0.
    pub fn set_bool(&mut self, name: &str, value: bool) -> Result<()> {
        let kind = self
            .layout
            .field(name)
            .map(|f| f.kind)
            .ok_or_else(|| LumaError::Uniform(format!("no field `{name}`")))?;
        match kind {
            UniformKind::F32 => self.set_f32(name, if value { 1.0 } else { 0.0 }),
            UniformKind::I32 => self.set_i32(name, value as i32),
            UniformKind::U32 | UniformKind::Bool => self.write(
                name,
                &(value as u32).to_le_bytes(),
                &[UniformKind::U32, UniformKind::Bool],
            ),
            other => Err(LumaError::Uniform(format!(
                "field `{name}` has kind {other:?}, cannot write bool"
            ))),
        }
    }

    pub fn set_vec2(&mut self, name: &str, value: Vec2) -> Result<()> {
        self.write(name, bytemuck::bytes_of(&value.to_array()), &[UniformKind::Vec2F])
    }

    pub fn set_vec3(&mut self, name: &str, value: Vec3) -> Result<()> {
        self.write(name, bytemuck::bytes_of(&value.to_array()), &[UniformKind::Vec3F])
    }

    pub fn set_vec4(&mut self, name: &str, value: Vec4) -> Result<()> {
        self.write(name, bytemuck::bytes_of(&value.to_array()), &[UniformKind::Vec4F])
    }

    /// Column-major 4x4 matrix.
    pub fn set_mat4(&mut self, name: &str, value: Mat4) -> Result<()> {
        self.write(
            name,
            bytemuck::bytes_of(&value.to_cols_array()),
            &[UniformKind::Mat4F],
        )
    }

    /// Column-major 3x3 matrix, padded to WGSL's 16-byte column stride.
    pub fn set_mat3(&mut self, name: &str, value: Mat3) -> Result<()> {
        let mut padded = [0f32; 12];
        let cols = value.to_cols_array();
        for c in 0..3 {
            padded[c * 4..c * 4 + 3].copy_from_slice(&cols[c * 3..c * 3 + 3]);
        }
        self.write(name, bytemuck::bytes_of(&padded), &[UniformKind::Mat3F])
    }

    /// Raw write for nested structs and arrays.
    pub fn set_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.write(name, bytes, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_moves_to_the_target_over_the_duration() {
        let mut animated = Animated::new(UniformTransition::Interpolation {
            duration: 1.0,
            easing: |t| t,
        });
        assert_eq!(animated.advance(&[0.0], 0.0), [0.0]);
        assert!(!animated.active());
        assert_eq!(animated.advance(&[2.0], 0.5), [0.0]);
        assert!(animated.active());
        assert_eq!(animated.advance(&[2.0], 1.0), [1.0]);
        assert_eq!(animated.advance(&[2.0], 1.5), [2.0]);
        assert!(!animated.active());
        // a new target during the run starts from the current value
        assert_eq!(animated.advance(&[4.0], 2.0), [2.0]);
        assert_eq!(animated.advance(&[4.0], 2.5), [3.0]);
    }

    #[test]
    fn spring_settles_on_the_target() {
        let mut animated = Animated::new(UniformTransition::Spring {
            stiffness: 0.5,
            damping: 0.5,
        });
        animated.advance(&[0.0, 0.0], 0.0);
        let mut value = animated.advance(&[1.0, -1.0], 1.0);
        assert!(value[0] > 0.0 && value[0] < 1.0);
        for frame in 2..200 {
            value = animated.advance(&[1.0, -1.0], frame as f64);
            if !animated.active() {
                break;
            }
        }
        assert!(!animated.active(), "settled");
        assert_eq!(value, [1.0, -1.0]);
    }
}
