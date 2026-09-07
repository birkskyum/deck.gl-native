//! Port of `@deck.gl/core/src/lib/layer-extension.ts`.
//!
//! An extension adds shader modules, code at the shader hooks (see [`crate::shaderlib::HOOKS`]),
//! vertex attributes and varyings to a layer, and fills its own uniforms before every draw.
//! Layers keep their extensions in [`LayerProps::extensions`](crate::LayerProps::extensions) and
//! hand them to the [`Extensions`] helpers when they assemble shaders, build attribute buffers
//! and update uniforms.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use luma_gl::shader::{assemble, ShaderAssembly, ShaderField, ShaderInjection};
use luma_gl::{AssembledShader, Model, ShaderModuleSource};
use wgpu::VertexFormat;

use crate::attribute_manager::{AttributeSource, BufferSpec};
use crate::layer::LayerContext;
use crate::shaderlib::HOOKS;
use crate::viewport::Viewport;
use crate::{DeckError, Result};

/// A vertex attribute an extension adds to the layer, one value per data row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionAttribute {
    pub name: &'static str,
    pub format: VertexFormat,
}

/// What an extension contributes to a layer's shader. Mirrors `LayerExtension.getShaders`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtensionShaders {
    pub modules: Vec<ShaderModuleSource>,
    pub injections: Vec<ShaderInjection>,
    pub attributes: Vec<ExtensionAttribute>,
    pub varyings: Vec<ShaderField>,
}

/// A layer extension. Implementations are immutable value objects: changing an option means
/// giving the layer a new instance, which rebuilds its model.
pub trait LayerExtension: Send + Sync + fmt::Debug {
    /// deck.gl's `extensionName`, for errors and logs.
    fn name(&self) -> &'static str;

    /// Shader code the extension adds to the layer.
    fn shaders(&self) -> ExtensionShaders;

    /// Values for the attributes declared in [`LayerExtension::shaders`], resolved against the
    /// layer's data like the layer's own accessors.
    fn attributes(&self) -> Vec<(&'static str, AttributeSource)> {
        Vec::new()
    }

    /// Write the extension's uniforms. Called on every layer update, before the upload.
    fn update_uniforms(&self, _model: &mut Model, _ctx: &LayerContext, _viewport: &Viewport) -> Result<()> {
        Ok(())
    }

    /// deck.gl's `equals`: same extension type with the same options. Implement with
    /// [`same_extension`].
    fn equals(&self, other: &dyn LayerExtension) -> bool;

    fn as_any(&self) -> &dyn Any;
}

/// True when `other` is a `T` equal to `this`. The usual body of [`LayerExtension::equals`].
pub fn same_extension<T: PartialEq + 'static>(this: &T, other: &dyn LayerExtension) -> bool {
    other.as_any().downcast_ref::<T>().is_some_and(|o| o == this)
}

/// The extensions of one layer, deck.gl's `extensions` prop.
#[derive(Clone, Default)]
pub struct Extensions(pub Vec<Arc<dyn LayerExtension>>);

impl Extensions {
    pub fn new(extensions: Vec<Arc<dyn LayerExtension>>) -> Self {
        Self(extensions)
    }

    pub fn from_one(extension: impl LayerExtension + 'static) -> Self {
        Self(vec![Arc::new(extension)])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn LayerExtension>> {
        self.0.iter()
    }

    /// The first extension of type `T`, if any.
    pub fn get<T: LayerExtension + 'static>(&self) -> Option<&T> {
        self.0.iter().find_map(|e| e.as_any().downcast_ref::<T>())
    }

    /// The combined shader contributions. Modules with the same name are only added once.
    pub fn shaders(&self) -> ExtensionShaders {
        let mut all = ExtensionShaders::default();
        for extension in &self.0 {
            let shaders = extension.shaders();
            for module in shaders.modules {
                if !all.modules.iter().any(|m| m.name == module.name) {
                    all.modules.push(module);
                }
            }
            all.injections.extend(shaders.injections);
            all.attributes.extend(shaders.attributes);
            all.varyings.extend(shaders.varyings);
        }
        all
    }

    /// Assemble a layer shader with deck.gl's hooks and everything the extensions add.
    pub fn assemble(
        &self,
        label: &str,
        modules: &[ShaderModuleSource],
        main: &str,
    ) -> Result<AssembledShader> {
        self.assemble_all(label, modules, main, self.shaders())
    }

    /// [`Extensions::assemble`] for layers that build their vertex buffers by hand and cannot
    /// bind extension attributes yet: fails with a clear message instead of a pipeline error.
    pub fn assemble_without_attributes(
        &self,
        label: &str,
        modules: &[ShaderModuleSource],
        main: &str,
    ) -> Result<AssembledShader> {
        self.assemble_own(label, modules, main, &ExtensionShaders::default())
    }

    /// [`Extensions::assemble_without_attributes`] plus `own`, shader contributions of the
    /// layer itself (a layer built on another layer's shader, like the trips layer on the path
    /// shader). Attributes in `own` are allowed: the layer binds their buffers.
    pub fn assemble_own(
        &self,
        label: &str,
        modules: &[ShaderModuleSource],
        main: &str,
        own: &ExtensionShaders,
    ) -> Result<AssembledShader> {
        let mut shaders = self.shaders();
        if let Some(attribute) = shaders.attributes.first() {
            return Err(DeckError::Layer {
                layer: label.to_string(),
                message: format!(
                    "extension attribute `{}` is not supported by this layer yet",
                    attribute.name
                ),
            });
        }
        for module in &own.modules {
            if !shaders.modules.iter().any(|m| m.name == module.name) {
                shaders.modules.push(*module);
            }
        }
        shaders.injections.extend(own.injections.iter().cloned());
        shaders.attributes.extend(own.attributes.iter().copied());
        shaders.varyings.extend(own.varyings.iter().copied());
        self.assemble_all(label, modules, main, shaders)
    }

    fn assemble_all(
        &self,
        label: &str,
        modules: &[ShaderModuleSource],
        main: &str,
        shaders: ExtensionShaders,
    ) -> Result<AssembledShader> {
        let mut all_modules: Vec<ShaderModuleSource> = modules.to_vec();
        for module in shaders.modules {
            if !all_modules.iter().any(|m| m.name == module.name) {
                all_modules.push(module);
            }
        }
        let attributes = shaders
            .attributes
            .iter()
            .map(|a| {
                Ok(ShaderField {
                    name: a.name,
                    ty: wgsl_type(a.format, label)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(assemble(&ShaderAssembly {
            label,
            modules: &all_modules,
            main,
            hooks: &HOOKS,
            injections: &shaders.injections,
            attributes: &attributes,
            varyings: &shaders.varyings,
        })?)
    }

    /// One instance buffer per extension attribute, at the locations the assembler chose.
    pub fn buffer_specs(&self, shader: &AssembledShader) -> Result<Vec<BufferSpec>> {
        self.shaders()
            .attributes
            .iter()
            .map(|a| {
                let location = shader
                    .attribute_location(a.name)
                    .ok_or_else(|| DeckError::Layer {
                        layer: shader.label.clone(),
                        message: format!(
                            "extension attribute `{}` was not assembled into the shader",
                            a.name
                        ),
                    })?;
                Ok(BufferSpec::instance(a.name, a.name, location, a.format))
            })
            .collect()
    }

    /// The attribute values of every extension, to append to the layer's own sources.
    pub fn sources(&self) -> Vec<(&'static str, AttributeSource)> {
        self.0.iter().flat_map(|e| e.attributes()).collect()
    }

    pub fn update_uniforms(&self, model: &mut Model, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        for extension in &self.0 {
            extension.update_uniforms(model, ctx, viewport)?;
        }
        Ok(())
    }
}

impl PartialEq for Extensions {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(a, b)| Arc::ptr_eq(a, b) || a.equals(b.as_ref()))
    }
}

impl fmt::Debug for Extensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter().map(|e| e.name())).finish()
    }
}

/// The WGSL type a vertex attribute of `format` is read as.
pub fn wgsl_type(format: VertexFormat, label: &str) -> Result<&'static str> {
    use VertexFormat::*;
    Ok(match format {
        Float32 | Unorm8 | Snorm8 | Unorm16 | Snorm16 | Float16 => "f32",
        Float32x2 | Unorm8x2 | Snorm8x2 | Unorm16x2 | Snorm16x2 | Float16x2 => "vec2<f32>",
        Float32x3 => "vec3<f32>",
        Float32x4 | Unorm8x4 | Snorm8x4 | Unorm16x4 | Snorm16x4 | Float16x4 | Unorm10_10_10_2
        | Unorm8x4Bgra => "vec4<f32>",
        Uint32 | Uint8 | Uint16 => "u32",
        Uint32x2 | Uint8x2 | Uint16x2 => "vec2<u32>",
        Uint32x3 => "vec3<u32>",
        Uint32x4 | Uint8x4 | Uint16x4 => "vec4<u32>",
        Sint32 | Sint8 | Sint16 => "i32",
        Sint32x2 | Sint8x2 | Sint16x2 => "vec2<i32>",
        Sint32x3 => "vec3<i32>",
        Sint32x4 | Sint8x4 | Sint16x4 => "vec4<i32>",
        other => {
            return Err(DeckError::Layer {
                layer: label.to_string(),
                message: format!("unsupported extension attribute format {other:?}"),
            })
        }
    })
}
