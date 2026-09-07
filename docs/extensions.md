# Shader hooks and layer extensions

deck.gl extensions (`DataFilterExtension`, `BrushingExtension`, ...) work by injecting code
into the layer shaders at fixed hooks and by adding shader modules, attributes and uniforms.
deck.gl's GLSL hooks (`DECKGL_FILTER_SIZE`, `DECKGL_FILTER_GL_POSITION`, `DECKGL_FILTER_COLOR`)
rely on `inout` parameters, function overloading and a preprocessor, none of which WGSL has,
and deck.gl JS does not support hooks in its WGSL path yet. The native port therefore defines
its own WGSL convention. It is implemented by the shader assembler in `luma-gl`
(`crates/luma-gl/src/shader.rs`) and by `deck_gl::extension`.

## The hooks

Every layer shader calls these functions, which the assembler generates:

| Hook key                       | Generated function                                                              |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `vs:DECKGL_FILTER_SIZE`        | `fn deckgl_filter_size(size: vec3<f32>, geometry: Geometry) -> vec3<f32>`       |
| `vs:DECKGL_FILTER_GL_POSITION` | `fn deckgl_filter_gl_position(position: vec4<f32>, geometry: Geometry) -> vec4<f32>` |
| `vs:DECKGL_FILTER_COLOR`       | `fn deckgl_filter_color(color: vec4<f32>, geometry: Geometry) -> vec4<f32>`     |
| `fs:DECKGL_FILTER_COLOR`       | `fn deckgl_filter_fragment_color(color: vec4<f32>, geometry: FragmentGeometry) -> vec4<f32>` |

The value is passed in and returned instead of `inout`. Inside injected code it is a `var`
with the name in the table (`size`, `position`, `color`) that the injection reads and assigns.
`geometry` is deck.gl's vertex geometry (`worldPosition`, `position` in common space, `uv`,
`normal`, `pickingColor`) or, in the fragment stage, the `FragmentGeometry` with `uv`.

Layer shaders call the hooks at the same places deck.gl's GLSL does: the size hook on the
offset of the vertex from the object's anchor, the position hook on the clip space position,
the colour hook on every colour that reaches the fragment stage, and the fragment colour hook
on the final colour before picking and highlighting.

## Attributes and varyings

WGSL entry points read attributes from an `Attributes` struct and pass values between the
stages through a `Varyings` struct, so an extension cannot simply declare `in` and `out`
globals. Instead it declares them as `ShaderField`s and the assembler:

- appends them to the layer's `Attributes` and `Varyings` structs at the next free
  `@location` (integer varyings get `@interpolate(flat)`),
- mirrors each one in a module scope `var<private>` of the same name, which is what injected
  code reads and writes,
- generates three functions the layer shaders call: `deckgl_vertex_start(attributes)` copies
  the attributes into the private variables, `deckgl_vertex_end(&varyings)` copies the
  varyings out, and `deckgl_fragment_start(varyings)` copies them back in.

The fixed injection points sit inside those functions: `vs:#main-start` at the end of
`deckgl_vertex_start`, `vs:#main-end` at the start of `deckgl_vertex_end` (where the layer's
own varyings are reachable through the `varyings` pointer, `(*varyings).vColor`), and
`fs:#main-start` at the end of `deckgl_fragment_start`. `vs:#decl` and `fs:#decl` land in
module scope. Uniforms need no special handling: a module declares
`@group(0) @binding(auto) var<uniform> name: Struct;` and the extension writes it through
`Model::uniforms("name")`.

## Writing an extension

Implement `deck_gl::LayerExtension`:

```rust
use deck_gl::luma_gl::{ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::{same_extension, Accessor, AttributeSource, ExtensionAttribute, ExtensionShaders, LayerExtension};

#[derive(Debug, PartialEq)]
struct TintExtension { get_tint: Accessor<f32>, scale: f32 }

const TINT_MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "tint",
    source: "struct TintUniforms { scale: f32 };\n@group(0) @binding(auto) var<uniform> tint: TintUniforms;\n",
};

impl LayerExtension for TintExtension {
    fn name(&self) -> &'static str { "TintExtension" }
    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders {
            modules: vec![TINT_MODULE],
            injections: vec![
                ShaderInjection::new("vs:#main-start", "tint_value = tintValues;"),
                ShaderInjection::new("fs:DECKGL_FILTER_COLOR",
                    "color = vec4<f32>(color.rgb * tint_value * tint.scale, color.a);"),
            ],
            attributes: vec![ExtensionAttribute { name: "tintValues", format: wgpu::VertexFormat::Float32 }],
            varyings: vec![ShaderField { name: "tint_value", ty: "f32" }],
        }
    }
    fn attributes(&self, _: &LayerData) -> deck_gl::Result<Vec<(&'static str, AttributeSource)>> {
        Ok(vec![("tintValues", AttributeSource::Floats(self.get_tint.clone()))])
    }
    fn update_uniforms(&self, model: &mut Model, _: &LayerContext, _: &Viewport, _: &LayerProps) -> deck_gl::Result<()> {
        model.uniforms("tint")?.set_f32("scale", self.scale)
    }
    fn equals(&self, other: &dyn LayerExtension) -> bool { same_extension(self, other) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}
```

and put it in the layer's base props: `LayerProps { extensions: Extensions::from_one(ext), ..LayerProps::new("id") }`.
Extension attributes are resolved against the layer's data like the layer's own accessors, so
they can be columns, constants or functions. Replacing an extension with one whose shader
contributions differ rebuilds the layer's model; changing only its accessors or options
re-uploads the attributes and uniforms.

Layers built on the `AttributeManager` (scatterplot, line, arc, point cloud, icon, column)
support extension attributes, and so do the path, polygon, trips, GeoJSON and cell layers,
which expand one value per object over their tessellated segments or vertices
(`AttributeManager::update_expanded`). The text, bitmap and screen grid layers support
modules, uniforms and injections, and return an error for extensions that declare
attributes.

## Built in extensions

`deck_gl_layers::extensions` ports `@deck.gl/extensions`:

- `DataFilterExtension`: hides objects whose one to four numeric values (`get_filter_value`,
  a `FilterValues`) fall outside `filter_range`, fades them between `filter_soft_range` and
  `filter_range` through their size and opacity, and keeps only the objects whose category keys
  (`get_filter_category`, a `FilterCategories` of small integers) are listed in
  `filter_categories`. `count_filtered` evaluates the same rules on the CPU, deck.gl's
  `onFilteredItemsChange` count. In JSON, `extensions: [{"@@type": "DataFilterExtension"}]`
  with the props on the layer, see [docs/json.md](json.md#extensions).
- `BrushingExtension`: shows only the objects within `brushing_radius` metres of the pointer,
  measured to the object's position, the other end of a line or arc, either end, or a custom
  position from `get_brushing_target`. The pointer is `LayerContext::pointer`, kept up to date
  by `Deck::pointer_move` and `Deck::pointer_leave` (and `deckgl_pointer_move` in the C API).
- `ClipExtension`: clips a layer to `clip_bounds` (`[left, bottom, right, top]` in the layer's
  coordinates), whole objects by their anchor with `clip_by_instance`, or the geometry per
  fragment without it.
- `MaskExtension`: keeps only what lies inside the geometry of the layer whose id is
  `mask_id`, or outside it with `mask_inverted`, by anchor (`mask_by_instance`) or per fragment.
  The mask layer has `operation: Operation::MASK`: the deck renders it into a 2048 pixel
  texture fitted to its bounds before the frame instead of on screen (deck.gl's `MaskEffect`),
  for up to four mask layers, with the map view. Layers report their bounds through
  `Layer::bounds`; without them the texture covers twice the view.
- `CollisionFilterExtension`: hides objects that overlap objects of a higher
  `get_collision_priority` (labels, icons). Before every frame the deck draws the layers of each
  `collision_group` into a half resolution map with their picking colours, sorted by priority
  (deck.gl's `CollisionFilterEffect`), and each object is kept only where the map shows it at
  its anchor, fading over a five pixel window. `collision_test_props` is not ported.
