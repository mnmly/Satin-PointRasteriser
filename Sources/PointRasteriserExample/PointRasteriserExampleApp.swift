#if os(macOS)
import Satin
import SwiftUI
import UniformTypeIdentifiers

@main
struct PointRasteriserExampleApp: App {
    private let renderer: PointRasteriserExampleRenderer

    init() {
        renderer = PointRasteriserExampleRenderer(
            initialCOPCURLs: Self.copcURLs(from: CommandLine.arguments)
        )
    }

    var body: some Scene {
        WindowGroup("Satin Point Rasteriser") {
            PointRasteriserContentView(renderer: renderer)
                .frame(minWidth: 640, minHeight: 640)
        }
        .windowResizability(.contentSize)
    }

    /// Every repeatable `--copc <path>` pair, resolved to a file URL. Paths
    /// that don't exist are skipped (one log line each) so a launch with
    /// `--copc a --copc b --copc c` degrades gracefully.
    private static func copcURLs(from arguments: [String]) -> [URL] {
        var urls: [URL] = []
        var index = 0
        while index < arguments.count {
            defer { index += 1 }
            guard arguments[index] == "--copc", arguments.indices.contains(index + 1) else { continue }
            index += 1
            let path = arguments[index]
            let url = path.hasPrefix("/")
                ? URL(fileURLWithPath: path)
                : URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: url.path) {
                urls.append(url)
            } else {
                print("[PointRasteriserExample] --copc: skipping missing file \(url.path)")
            }
        }
        return urls
    }
}

/// Top-level content: the Satin metal view plus a toolbar (Open PLY/COPC,
/// Settings) and a bottom status/error overlay, modeled on
/// Satin-ComputeRasteriser's `ComputeRasteriserAppView`.
private struct PointRasteriserContentView: View {
    let renderer: PointRasteriserExampleRenderer
    @State private var isPLYImporterPresented = false
    @State private var isCOPCImporterPresented = false
    @State private var isSettingsPresented = false

    var body: some View {
        NavigationStack {
            SatinMetalView(renderer: renderer)
                .ignoresSafeArea()
                .navigationTitle(renderer.appState.status)
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Menu {
                            Button {
                                isPLYImporterPresented = true
                            } label: {
                                Label("Open PLY…", systemImage: "doc")
                            }
                            #if canImport(SwiftPDAL)
                            Button {
                                isCOPCImporterPresented = true
                            } label: {
                                Label("Open COPC…", systemImage: "globe")
                            }
                            #endif
                        } label: {
                            Label("Open", systemImage: "plus")
                        }
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            isSettingsPresented = true
                        } label: {
                            Label("Settings", systemImage: "slider.horizontal.3")
                        }
                    }
                }
                .safeAreaInset(edge: .bottom) {
                    if renderer.appState.errorMessage != nil || renderer.appState.isLoading || renderer.appState.isStreaming {
                        statusOverlay
                            .padding(.horizontal)
                            .padding(.bottom, 8)
                    }
                }
        }
        .sheet(isPresented: $isSettingsPresented) {
            PointRasteriserSettingsView(appState: renderer.appState, renderer: renderer)
        }
        .fileImporter(
            isPresented: $isPLYImporterPresented,
            allowedContentTypes: [.ply],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case let .success(urls):
                if let url = urls.first {
                    renderer.loadPLY(url: url)
                }
            case let .failure(error):
                renderer.appState.errorMessage = error.localizedDescription
            }
        }
        #if canImport(SwiftPDAL)
        .fileImporter(
            isPresented: $isCOPCImporterPresented,
            allowedContentTypes: [.copcLAZ],
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case let .success(urls):
                if !urls.isEmpty {
                    renderer.loadCOPC(urls: urls)
                }
            case let .failure(error):
                renderer.appState.errorMessage = error.localizedDescription
            }
        }
        #endif
    }

    @ViewBuilder
    private var statusOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            if renderer.appState.isLoading {
                Label("Loading…", systemImage: "hourglass")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let error = renderer.appState.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(3)
            }
            #if canImport(SwiftPDAL)
            if renderer.appState.isStreaming {
                Text("\(renderer.appState.streamingChunks) chunks (\(renderer.appState.streamingPinnedChunks) coarse-pinned) · \(renderer.appState.streamingPoints) pts · \(renderer.appState.streamingFreeSlots) free")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                Text(String(format: "decode %.1f M pts/s · %d pending · %d starved", renderer.appState.streamingDecodeMPS, renderer.appState.streamingPendingUploads, renderer.appState.streamingStarvedTicks))
                    .font(.caption2)
                    .foregroundStyle(renderer.appState.streamingStarvedTicks > 0 ? .orange : .secondary)
                    .monospacedDigit()
            }
            #endif
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

extension UTType {
    static let ply = UTType(filenameExtension: "ply") ?? .data
    /// COPC files use the LAZ extension. The fileImporter shouldn't reject
    /// `.las` or non-COPC `.laz` either — SwiftPDAL's open call surfaces the
    /// "not a COPC" error through the standard error path.
    static let copcLAZ = UTType(filenameExtension: "laz") ?? .data
}
#else
// The example app is macOS-only (SwiftUI + AppKit-backed SatinMetalView).
@main
struct PointRasteriserExampleApp {
    static func main() {
        print("PointRasteriserExample is only available on macOS.")
    }
}
#endif
