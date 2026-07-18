// Composite of the resolved LOD point cloud over the scene. Ported from
// Satin-ComputeRasteriser's Post.metal. Two variants, selected by the
// SourceMaterial's label (which drives the `<label>Vertex/Fragment` function
// names): a plain always-on-top blend, and a depth-aware blend that writes the
// cloud's reverse-Z NDC depth so regular Satin meshes inter-occlude with it.
//
// A depth vertex twin (`pointRasteriserPostDepthVertex`) is provided so the
// depth material resolves its own `<label>Vertex` name — the two vertex
// functions are identical full-screen-quad pass-throughs.

vertex VertexData pointRasteriserPostVertex(
    Vertex in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant VertexUniforms *vertexUniforms [[buffer(VertexBufferVertexUniforms)]]
) {
    VertexData out;
    out.position = float4(in.position, 1.0);
    out.texcoord = in.texcoord;
    return out;
}

vertex VertexData pointRasteriserPostDepthVertex(
    Vertex in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant VertexUniforms *vertexUniforms [[buffer(VertexBufferVertexUniforms)]]
) {
    VertexData out;
    out.position = float4(in.position, 1.0);
    out.texcoord = in.texcoord;
    return out;
}

// Legacy composite: blends the resolved cloud color over the target with no
// depth interaction (always on top). Used when `writesSceneDepth` is false or
// the render pass has no depth attachment.
fragment float4 pointRasteriserPostFragment(
    VertexData in [[stage_in]],
    texture2d<float> resultTexture [[texture(FragmentTextureCustom1)]]
) {
    constexpr sampler s(filter::linear, mip_filter::nearest);
    return resultTexture.sample(s, in.texcoord);
}

// Depth-aware composite: outputs the cloud's per-pixel reversed-Z NDC depth so
// regular Satin meshes correctly inter-occlude with the cloud. The depth
// texture stores 0 for pixels with no cloud (far in reversed-Z), discarded so
// the composite never touches background / mesh-only pixels.
struct PointRasteriserPostOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment PointRasteriserPostOut pointRasteriserPostDepthFragment(
    VertexData in [[stage_in]],
    texture2d<float> resultTexture [[texture(FragmentTextureCustom1)]],
    texture2d<float> depthTexture [[texture(FragmentTextureCustom2)]]
) {
    constexpr sampler colorSampler(filter::linear, mip_filter::nearest);
    constexpr sampler depthSampler(filter::nearest, mip_filter::nearest);
    const float d = depthTexture.sample(depthSampler, in.texcoord).r;
    if (d <= 0.0) {
        discard_fragment();
    }
    PointRasteriserPostOut out;
    out.color = resultTexture.sample(colorSampler, in.texcoord);
    out.depth = d;
    return out;
}
