import Cocoa

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var label: NSTextField!
    var process: Process?
    var buffer = ""
    var ready = false
    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let appItem = NSMenuItem(); menu.addItem(appItem)
        let appMenu = NSMenu(); appMenu.addItem(withTitle: "Quit M5Diffusion", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu; NSApp.mainMenu = menu
        window = NSWindow(contentRect: NSRect(x:0,y:0,width:520,height:250), styleMask:[.titled,.closable,.miniaturizable], backing:.buffered, defer:false)
        window.title = "M5Diffusion v0.1.0-alpha"; window.center()
        label = NSTextField(wrappingLabelWithString:"画像スタジオを準備しています…")
        label.frame = NSRect(x:28,y:90,width:464,height:115); label.font = .systemFont(ofSize:15)
        window.contentView?.addSubview(label)
        let open = NSButton(title:"画面を開く", target:self, action:#selector(openUI)); open.frame=NSRect(x:28,y:28,width:150,height:36); window.contentView?.addSubview(open)
        let quit = NSButton(title:"アプリを終了", target:NSApp, action:#selector(NSApplication.terminate(_:))); quit.frame=NSRect(x:338,y:28,width:150,height:36); window.contentView?.addSubview(quit)
        window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps:true)
        guard let resources = Bundle.main.resourceURL else { label.stringValue="アプリを入れ直してください。"; return }
        let child=Process(); process=child; child.executableURL=resources.appendingPathComponent("runtime/bin/python3")
        child.arguments=["-I","-B",resources.appendingPathComponent("bootstrap.py").path]
        child.environment=ProcessInfo.processInfo.environment.merging(["PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1"]) { _,new in new }
        let pipe=Pipe(); child.standardOutput=pipe; child.standardError=FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data=handle.availableData
            guard !data.isEmpty, let text=String(data:data,encoding:.utf8) else { return }
            DispatchQueue.main.async { self?.receive(text) }
        }
        child.terminationHandler = { [weak self] task in
            DispatchQueue.main.async { if task.terminationStatus != 0 && self?.ready == true { self?.label.stringValue="画像処理が終了しました。アプリを開き直してください。" } }
        }
        do { try child.run() } catch { label.stringValue="アプリ内の実行環境を起動できません。アプリを入れ直してください。" }
    }
    func receive(_ text:String) {
        buffer += text
        while let end=buffer.firstIndex(of:"\n") {
            let line=String(buffer[..<end]); buffer=String(buffer[buffer.index(after:end)...])
            if line.hasPrefix("READY:") { ready=true; label.stringValue="起動しました。画像を生成、または高画質化できます。\n終了ボタンで画像処理も停止します。"; openUI() }
            else if line.hasPrefix("ERROR:") { label.stringValue=String(line.dropFirst(6)) }
            else if line.hasPrefix("STATUS:") { label.stringValue=String(line.dropFirst(7)) }
        }
    }
    @objc func openUI() { if ready { NSWorkspace.shared.open(URL(string:"http://127.0.0.1:7861/")!) } }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool { true }
    func applicationWillTerminate(_ notification:Notification) { if let process=process, process.isRunning { process.terminate(); process.waitUntilExit() } }
}
let application=NSApplication.shared
let delegate=AppDelegate()
application.delegate=delegate
application.setActivationPolicy(.regular)
application.run()
