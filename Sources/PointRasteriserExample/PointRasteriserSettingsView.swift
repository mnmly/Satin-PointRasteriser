#if os(macOS)
import AppKit
import SatinPointRasteriser
import SwiftUI

/// Settings sheet exposing every ``PointRasteriserConfiguration`` field (plus
/// the DoF recipe and LOD-sweep telemetry), modeled on Satin-ComputeRasteriser's
/// `SettingsSheet`. Bound directly to the renderer's `@Observable` ``PointRasteriserExampleState``
/// — see that type's doc comment for the (small) set of fields deliberately left out.
struct PointRasteriserSettingsView: View {
    @Bindable var appState: PointRasteriserExampleState
    let renderer: PointRasteriserExampleRenderer
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Render mode") {
                    Picker("Mode", selection: $appState.renderMode) {
                        Text("High Quality Average").tag(RenderMode.highQualityAverage)
                        Text("Nearest Point").tag(RenderMode.nearestPoint)
                    }
                    .pickerStyle(.segmented)

                    Toggle("SIMD-group aggregation", isOn: $appState.enableSimdAggregation)
                }

                Section("Point sizing") {
                    Picker("Mode", selection: $appState.pointSizeMode) {
                        Text("Screen").tag(PointSizeMode.screenSpace)
                        Text("World").tag(PointSizeMode.worldSpace)
                    }
                    .pickerStyle(.segmented)

                    sliderRow(title: "Minimum", value: floatBinding(\.minimumPointSize), range: 1 ... 32,
                              formatter: { String(format: "%.0f", $0) })
                    sliderRow(title: "Maximum", value: floatBinding(\.maximumPointSize), range: 1 ... 128,
                              formatter: { String(format: "%.0f", $0) })
                    sliderRow(
                        title: "Scale",
                        value: floatBinding(\.pointSizeScale),
                        range: appState.pointSizeMode == .worldSpace ? 0.001 ... 0.1 : 1 ... 16,
                        formatter: { String(format: appState.pointSizeMode == .worldSpace ? "%.3f" : "%.1f", $0) }
                    )
                }

                Section("LOD & culling") {
                    Toggle("Frustum culling", isOn: $appState.enableFrustumCulling)
                    Toggle("Continuous LOD", isOn: $appState.enableCLOD)
                    Stepper("LOD bias: \(appState.lodBias)", value: $appState.lodBias, in: -3 ... 7)
                    Toggle("LOD dither", isOn: $appState.enableLODDither)
                        .disabled(!appState.enableCLOD)
                }

                Section("Amortized LOD (double buffering)") {
                    Stepper(
                        "Budget: \(appState.lodPointsPerFrame == 0 ? "full sweep" : "\(appState.lodPointsPerFrame) pts/frame")",
                        value: $appState.lodPointsPerFrame,
                        in: 0 ... 2_000_000,
                        step: 5_000
                    )
                    Button("Restart sweep") { renderer.restartLODSweep() }
                    LabeledContent("Sweep progress") {
                        Text("\(Int(appState.lodSweepProgress * 100))%")
                            .monospacedDigit()
                    }
                    LabeledContent("LOD count") {
                        Text("\(appState.lodCount)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("LODSelect") {
                        Text(appState.lodSelectSkipped ? "skipped (static)" : "ran")
                            .monospacedDigit()
                            .foregroundStyle(appState.lodSelectSkipped ? .green : .secondary)
                    }
                    if appState.lodOverflowed {
                        Label("LOD capacity exceeded — dropped \(appState.lodOverflow) points; increase lodCapacity",
                              systemImage: "exclamationmark.triangle.fill")
                            .font(.callout.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.red, in: RoundedRectangle(cornerRadius: 6))
                            .listRowInsets(EdgeInsets())
                    }
                }

                Section("Rejection & resolve") {
                    Toggle("Point rejection (VAST cone)", isOn: $appState.enablePointRejection)
                    sliderRow(title: "Cone threshold", value: floatBinding(\.rejectionConeThreshold),
                              range: 0 ... (.pi / 2), formatter: { String(format: "%.2f rad", $0) })
                        .disabled(!appState.enablePointRejection)
                    sliderRow(title: "Depth tolerance", value: floatBinding(\.depthTolerance),
                              range: 0 ... 0.1, formatter: { String(format: "%.3f", $0) })
                    Stepper("Hole fill: \(appState.holeFillIterations)", value: $appState.holeFillIterations, in: 0 ... 4)
                }

                Section("Rendering") {
                    Toggle("Write scene depth", isOn: $appState.writesSceneDepth)
                    Toggle("Colorize chunks", isOn: $appState.colorizeChunks)
                    Toggle("Colorize overdraw", isOn: $appState.colorizeOverdraw)
                    ColorPicker("Background", selection: backgroundColorBinding, supportsOpacity: true)
                }

                Section("Motion blur") {
                    sliderRow(title: "Shutter strength", value: floatBinding(\.motionBlur), range: 0 ... 1,
                              formatter: { String(format: "%.2f", $0) })
                    Stepper("Samples: \(appState.motionBlurSamples)", value: $appState.motionBlurSamples, in: 1 ... 16)
                        .disabled(appState.motionBlur <= 0)
                    sliderRow(title: "Max spread (px)", value: floatBinding(\.motionBlurMaxSpread), range: 1 ... 256,
                              formatter: { String(format: "%.0f", $0) })
                        .disabled(appState.motionBlur <= 0)
                }

                Section("Sine-wave displacement demo") {
                    Toggle("Enable ('D' key)", isOn: $appState.sineDisplacementEnabled)
                    sliderRow(title: "Amplitude (× cloud radius)", value: floatBinding(\.sineDisplacementAmplitude), range: 0 ... 0.5,
                              formatter: { String(format: "%.3f", $0) })
                        .disabled(!appState.sineDisplacementEnabled)
                    sliderRow(title: "Frequency (cycles / radius)", value: floatBinding(\.sineDisplacementFrequency), range: 0 ... 30,
                              formatter: { String(format: "%.1f", $0) })
                        .disabled(!appState.sineDisplacementEnabled)
                }

                #if canImport(SwiftPDAL)
                if appState.isStreaming {
                    Section("Streaming (COPC)") {
                        LabeledContent("Resident") {
                            Text("\(appState.streamingChunks) chunks · \(appState.streamingPoints) pts")
                                .monospacedDigit()
                        }
                        LabeledContent("Coarse pinned") {
                            Text("\(appState.streamingPinnedChunks) (guaranteed coverage)")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        LabeledContent("Decode rate") {
                            Text(String(format: "%.1f M pts/s", appState.streamingDecodeMPS))
                                .monospacedDigit()
                        }
                        LabeledContent("Decode queue") {
                            Text("\(appState.streamingDecodePending) pending · \(appState.streamingDecodeInFlight) in-flight")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        LabeledContent("Upload / starved") {
                            Text("\(appState.streamingPendingUploads) / \(appState.streamingStarvedTicks)")
                                .monospacedDigit()
                                .foregroundStyle(appState.streamingStarvedTicks > 0 ? .orange : .secondary)
                        }
                        LabeledContent("Free slots") {
                            Text("\(appState.streamingFreeSlots)")
                                .monospacedDigit()
                        }
                        sliderRow(
                            title: "Detail (chunk px)",
                            value: Binding(
                                get: { Double(appState.streamingTargetChunkPx) },
                                set: { renderer.setStreamingTargetChunkPx(Float($0)) }
                            ),
                            range: 32 ... 512,
                            step: 16,
                            formatter: { "\(Int($0)) px" }
                        )
                        sliderRow(
                            title: "Budget",
                            value: Binding(
                                get: { Double(appState.streamingBudgetMB) },
                                set: { renderer.setStreamingBudget(MB: Int($0)) }
                            ),
                            range: 256 ... 16384,
                            step: 256,
                            formatter: { "\(Int($0)) MB" }
                        )
                        Picker("Residency", selection: Binding(
                            get: { appState.streamingResidency },
                            set: { renderer.setResidency($0) }
                        )) {
                            Text("Halo").tag(StreamingResidencyChoice.halo)
                            Text("Distance").tag(StreamingResidencyChoice.distance)
                        }
                        .pickerStyle(.segmented)
                    }
                }
                #endif

                // Depth of field — translucent defocus (weighted-blended OIT) plus a
                // jitter spread. Out-of-focus points become see-through and scatter,
                // so they blend instead of hard-occluding. The band/falloff are
                // fractions of the focal distance, so they auto-scale to any cloud.
                Section("Depth of field") {
                    Toggle("Enable", isOn: $appState.dofEnabled)
                    if appState.dofEnabled {
                        Toggle("Translucent (OIT)", isOn: $appState.dofTranslucent)
                        Toggle("Jitter spread", isOn: $appState.dofJitter)
                        Toggle("Auto focus (cloud centre)", isOn: $appState.dofAutoFocus)
                        if !appState.dofAutoFocus {
                            sliderRow(title: "Focus", value: floatBinding(\.dofFocus),
                                      range: 0 ... Double(max(appState.dofFocusMax, 0.01)),
                                      formatter: { String(format: "%.2f", $0) })
                        }
                        sliderRow(title: "Band", value: floatBinding(\.dofBand),
                                  range: 0 ... 0.5, formatter: { String(format: "%.3f", $0) })
                        sliderRow(title: "Falloff", value: floatBinding(\.dofFalloff),
                                  range: 0.01 ... 1.0, formatter: { String(format: "%.3f", $0) })
                        sliderRow(title: "Scatter", value: floatBinding(\.dofScatter),
                                  range: 0 ... 0.4, formatter: { String(format: "%.3f", $0) })
                            .disabled(!appState.dofJitter)
                        sliderRow(title: "Max defocus", value: floatBinding(\.dofMaxDefocus),
                                  range: 0 ... 1.0, formatter: { String(format: "%.2f", $0) })
                            .disabled(!appState.dofTranslucent)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .frame(minWidth: 380, minHeight: 480)
    }

    /// Bridge a `Float` app-state property to the `Binding<Double>` the sliders use.
    private func floatBinding(_ keyPath: ReferenceWritableKeyPath<PointRasteriserExampleState, Float>) -> Binding<Double> {
        Binding(
            get: { Double(appState[keyPath: keyPath]) },
            set: { appState[keyPath: keyPath] = Float($0) }
        )
    }

    private var backgroundColorBinding: Binding<Color> {
        Binding(
            get: {
                let c = appState.backgroundColor
                return Color(.sRGB, red: Double(c.x), green: Double(c.y), blue: Double(c.z), opacity: Double(c.w))
            },
            set: { newColor in
                let ns = NSColor(newColor).usingColorSpace(.sRGB) ?? NSColor(newColor)
                appState.backgroundColor = SIMD4<Float>(
                    Float(ns.redComponent), Float(ns.greenComponent), Float(ns.blueComponent), Float(ns.alphaComponent)
                )
            }
        )
    }

    @ViewBuilder
    private func sliderRow(
        title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double? = nil,
        formatter: @escaping (Double) -> String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(formatter(value.wrappedValue))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            if let step {
                Slider(value: value, in: range, step: step)
            } else {
                Slider(value: value, in: range)
            }
        }
    }
}
#endif
