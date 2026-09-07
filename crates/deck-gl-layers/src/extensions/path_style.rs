//! Port of `@deck.gl/extensions/src/path-style`: dashed and offset paths, and dashed
//! scatterplot strokes.

use std::any::Any;

use deck_gl::attribute_manager::AttributeSource;
use deck_gl::luma_gl::{Model, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::{
    same_extension, Accessor, ExtensionAttribute, ExtensionShaders, LayerContext, LayerData, LayerExtension,
    LayerProps, Result, Viewport,
};
use wgpu::VertexFormat;

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "pathStyle",
    source: include_str!("../wgsl/path_style.wgsl"),
};

/// The layer the extension is attached to: deck.gl deduces it from the layer, here it is
/// chosen up front (the JSON layers do it from the layer type).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PathStyleTarget {
    /// Path, trips and the strokes of polygon layers: dashes and offsets
    #[default]
    Path,
    /// Scatterplot strokes: dashes along the circle
    Scatterplot,
}

/// deck.gl's `PathStyleExtension`: with the `dash` option paths are drawn dashed by
/// `get_dash_array` (`[solid, gap]` in path widths), with `offset` they shift sideways by
/// `get_offset` widths. `high_precision_dash` is not ported: dashes restart at every segment.
#[derive(Clone, Debug, PartialEq)]
pub struct PathStyleExtension {
    pub dash: bool,
    pub offset: bool,
    pub target: PathStyleTarget,
    pub get_dash_array: Accessor<[f32; 2]>,
    pub get_offset: Accessor<f32>,
    /// Stretch the dashes so that both ends of a path end with half a dash
    pub dash_justified: bool,
    /// Whether the gaps between dashes can be picked
    pub dash_gap_pickable: bool,
}

impl Default for PathStyleExtension {
    fn default() -> Self {
        Self {
            dash: false,
            offset: false,
            target: PathStyleTarget::Path,
            get_dash_array: Accessor::Constant([0.0, 0.0]),
            get_offset: Accessor::Constant(0.0),
            dash_justified: false,
            dash_gap_pickable: false,
        }
    }
}

impl PathStyleExtension {
    /// Dashed paths.
    pub fn dashed(get_dash_array: Accessor<[f32; 2]>) -> Self {
        Self {
            dash: true,
            get_dash_array,
            ..Default::default()
        }
    }

    /// Offset paths.
    pub fn offset(get_offset: Accessor<f32>) -> Self {
        Self {
            offset: true,
            get_offset,
            ..Default::default()
        }
    }
}

const PATH_DASH_FS: &str = "let dash_solidLength = vDashArray.x;
  let dash_gapLength = vDashArray.y;
  var dash_unitLength = dash_solidLength + dash_gapLength;
  if (dash_unitLength > 0.0) {
    var dash_offset: f32;
    if (pathStyle.dashAlignMode == 0.0) {
      dash_offset = vDashOffset;
    } else {
      dash_unitLength = varyings.vPathLength / round(varyings.vPathLength / dash_unitLength);
      dash_offset = dash_solidLength / 2.0;
    }
    let dash_along = varyings.vPathPosition.y + dash_offset;
    let dash_unitOffset = dash_along - dash_unitLength * floor(dash_along / dash_unitLength);
    if (dash_gapLength > 0.0 && dash_unitOffset > dash_solidLength) {
      if (path.capType <= 0.5) {
        if (!(pathStyle.dashGapPickable != 0 && picking.isActive > 0.5)) {
          discard;
        }
      } else {
        // caps are rounded, test the distance to the solid ends
        let dash_distToEnd = length(vec2<f32>(
          min(dash_unitOffset - dash_solidLength, dash_unitLength - dash_unitOffset),
          varyings.vPathPosition.x
        ));
        if (dash_distToEnd > 1.0) {
          if (!(pathStyle.dashGapPickable != 0 && picking.isActive > 0.5)) {
            discard;
          }
        }
      }
    }
  }";

const PATH_OFFSET_SIZE: &str = "let offset_width = abs(instanceOffsets * 2.0) + 1.0;
  size = size * offset_width;";

const PATH_OFFSET_END: &str = "let offset_width = abs(instanceOffsets * 2.0) + 1.0;
  let offset_dir = sign(instanceOffsets);
  (*varyings).vPathPosition.x = ((*varyings).vPathPosition.x + offset_dir) * offset_width - offset_dir;
  (*varyings).vPathPosition.y = (*varyings).vPathPosition.y * offset_width;
  (*varyings).vPathLength = (*varyings).vPathLength * offset_width;";

const PATH_OFFSET_FS: &str =
    "let offset_isInside = step(-1.0, varyings.vPathPosition.x) * step(varyings.vPathPosition.x, 1.0);
  if (offset_isInside == 0.0) {
    discard;
  }";

const SCATTERPLOT_DASH_FS: &str = "let dash_unitLength = vDashArray.x + vDashArray.y;
  if (dash_unitLength > 0.0 && scatterplot.stroked > 0.5) {
    let dash_distToCenter = length(varyings.unitPosition) * varyings.outerRadiusPixels;
    let dash_innerRadius = varyings.innerUnitRadius * varyings.outerRadiusPixels;
    if (dash_distToCenter >= dash_innerRadius) {
      let dash_strokeWidth = (1.0 - varyings.innerUnitRadius) * varyings.outerRadiusPixels;
      let dash_midStrokeRadius = (varyings.innerUnitRadius + 1.0) * 0.5 * varyings.outerRadiusPixels;
      let dash_angle = atan2(varyings.unitPosition.y, varyings.unitPosition.x) + PI;
      let dash_circumference = 2.0 * PI * dash_midStrokeRadius;
      let dash_posAlongStroke = (dash_angle / (2.0 * PI)) * dash_circumference / dash_strokeWidth;
      let dash_unitOffset = dash_posAlongStroke - dash_unitLength * floor(dash_posAlongStroke / dash_unitLength);
      if (dash_unitOffset > vDashArray.x) {
        if (scatterplot.filled != 0) {
          dash_inGap = true;
          dash_fillColor = varyings.vFillColor;
          dash_lineColor = varyings.vLineColor;
        } else if (!(pathStyle.dashGapPickable != 0 && picking.isActive > 0.5)) {
          discard;
        }
      }
    }
  }";

const SCATTERPLOT_DASH_COLOR: &str = "if (dash_inGap) {
    let dash_alphaFactor = color.a / max(dash_lineColor.a, 0.001);
    color = vec4<f32>(dash_fillColor.rgb, dash_fillColor.a * dash_alphaFactor);
  }";

impl LayerExtension for PathStyleExtension {
    fn name(&self) -> &'static str {
        "PathStyleExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        let mut shaders = ExtensionShaders::default();
        match self.target {
            PathStyleTarget::Path => {
                if self.dash {
                    shaders.modules.push(MODULE);
                    shaders.injections.push(ShaderInjection::new(
                        "vs:#main-end",
                        "vDashArray = instanceDashArrays;\n  vDashOffset = 0.0;",
                    ));
                    shaders
                        .injections
                        .push(ShaderInjection::new("fs:#main-start", PATH_DASH_FS));
                    shaders.attributes.push(ExtensionAttribute {
                        name: "instanceDashArrays",
                        format: VertexFormat::Float32x2,
                    });
                    shaders.varyings.push(ShaderField {
                        name: "vDashArray",
                        ty: "vec2<f32>",
                    });
                    shaders.varyings.push(ShaderField {
                        name: "vDashOffset",
                        ty: "f32",
                    });
                }
                if self.offset {
                    shaders
                        .injections
                        .push(ShaderInjection::new("vs:DECKGL_FILTER_SIZE", PATH_OFFSET_SIZE));
                    shaders
                        .injections
                        .push(ShaderInjection::new("vs:#main-end", PATH_OFFSET_END).with_order(1));
                    shaders
                        .injections
                        .push(ShaderInjection::new("fs:#main-start", PATH_OFFSET_FS).with_order(1));
                    shaders.attributes.push(ExtensionAttribute {
                        name: "instanceOffsets",
                        format: VertexFormat::Float32,
                    });
                }
            }
            PathStyleTarget::Scatterplot => {
                if self.dash {
                    shaders.modules.push(MODULE);
                    shaders.injections.push(ShaderInjection::new(
                        "fs:#decl",
                        "var<private> dash_inGap: bool = false;\nvar<private> dash_fillColor: vec4<f32>;\nvar<private> dash_lineColor: vec4<f32>;",
                    ));
                    shaders.injections.push(ShaderInjection::new(
                        "vs:#main-end",
                        "vDashArray = instanceDashArrays;",
                    ));
                    shaders
                        .injections
                        .push(ShaderInjection::new("fs:#main-start", SCATTERPLOT_DASH_FS));
                    shaders.injections.push(ShaderInjection::new(
                        "fs:DECKGL_FILTER_COLOR",
                        SCATTERPLOT_DASH_COLOR,
                    ));
                    shaders.attributes.push(ExtensionAttribute {
                        name: "instanceDashArrays",
                        format: VertexFormat::Float32x2,
                    });
                    shaders.varyings.push(ShaderField {
                        name: "vDashArray",
                        ty: "vec2<f32>",
                    });
                }
            }
        }
        shaders
    }

    fn attributes(&self, _data: &LayerData) -> Result<Vec<(&'static str, AttributeSource)>> {
        let mut sources = Vec::new();
        if self.dash {
            sources.push((
                "instanceDashArrays",
                AttributeSource::Vec2(self.get_dash_array.clone()),
            ));
        }
        if self.offset && self.target == PathStyleTarget::Path {
            sources.push((
                "instanceOffsets",
                AttributeSource::Floats(self.get_offset.clone()),
            ));
        }
        Ok(sources)
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        _ctx: &LayerContext,
        _viewport: &Viewport,
        _props: &LayerProps,
    ) -> Result<()> {
        if !self.dash {
            return Ok(());
        }
        let u = model.uniforms("pathStyle")?;
        u.set_f32("dashAlignMode", if self.dash_justified { 1.0 } else { 0.0 })?;
        u.set_i32("dashGapPickable", self.dash_gap_pickable as i32)?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
