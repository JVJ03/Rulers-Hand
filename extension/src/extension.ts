// Claudelash — MILESTONE 3.
//
// Listens on localhost for gesture events from the Python watcher and logs
// them to an output channel. It deliberately does NOT touch any terminal yet:
// the point of this milestone is to prove the Python -> extension round-trip
// works before anything can send stray keystrokes anywhere.
//
// Local-only by design. The server binds 127.0.0.1 explicitly, so it is not
// reachable from the network even if the firewall would allow it.

import * as http from 'http';
import * as vscode from 'vscode';

interface GestureAction {
    action: string;
    key?: string;
    text?: string;
    description?: string;
}

interface GestureMap {
    gestures: Record<string, GestureAction>;
}

let output: vscode.OutputChannel;
let server: http.Server | undefined;
let gestureMap: GestureMap = { gestures: {} };
let received = 0;

const MAX_BODY_BYTES = 64 * 1024; // a gesture payload is a few hundred bytes

function log(message: string): void {
    const now = new Date().toISOString().slice(11, 23);
    output.appendLine(`[${now}] ${message}`);
}

async function loadGestureMap(context: vscode.ExtensionContext): Promise<void> {
    // Resolve against the extension's own URI rather than guessing a path —
    // this keeps working regardless of where the folder actually lives.
    const uri = vscode.Uri.joinPath(context.extensionUri, 'gesture-map.json');
    try {
        const bytes = await vscode.workspace.fs.readFile(uri);
        const parsed = JSON.parse(Buffer.from(bytes).toString('utf8')) as GestureMap;
        gestureMap = { gestures: parsed.gestures ?? {} };
        const names = Object.keys(gestureMap.gestures);
        log(`Gesture map loaded: ${names.length ? names.join(', ') : '(empty)'}`);
    } catch (err) {
        gestureMap = { gestures: {} };
        log(`Could not read gesture-map.json — ${describeError(err)}`);
    }
}

/** Never surface a raw error string; they can carry paths or internals. */
function describeError(err: unknown): string {
    if (err instanceof Error && err.name) {
        const known = ['SyntaxError', 'FileNotFound', 'EntryNotFound', 'NoPermissions'];
        return known.includes(err.name) ? err.name : 'unreadable';
    }
    return 'unreadable';
}

function handleGesture(payload: Record<string, unknown>): void {
    received += 1;
    const name = typeof payload.gesture === 'string' ? payload.gesture : '(missing)';
    const held = typeof payload.held_ms === 'number' ? `${payload.held_ms}ms` : '?';

    const mapped = gestureMap.gestures[name];
    if (mapped) {
        // Milestone 3 stops here. Milestone 4 replaces this line with the
        // actual terminal write.
        const detail = mapped.key ?? mapped.text ?? '';
        log(`#${received}  ${name}  held ${held}  -> would ${mapped.action}(${detail})`);
    } else {
        log(`#${received}  ${name}  held ${held}  -> no mapping, ignored`);
    }
}

function startServer(port: number): void {
    server = http.createServer((req, res) => {
        if (req.method !== 'POST' || req.url !== '/gesture') {
            res.writeHead(404).end();
            return;
        }

        let body = '';
        let tooBig = false;
        req.on('data', (chunk: Buffer) => {
            body += chunk.toString('utf8');
            if (body.length > MAX_BODY_BYTES) {
                tooBig = true;
                res.writeHead(413).end();
                req.destroy();
            }
        });
        req.on('end', () => {
            if (tooBig) {
                return;
            }
            try {
                handleGesture(JSON.parse(body) as Record<string, unknown>);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end('{"ok":true}');
            } catch {
                log('Received a POST whose body was not valid JSON — ignored');
                res.writeHead(400).end();
            }
        });
    });

    server.on('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EADDRINUSE') {
            log(`Port ${port} is already in use. Another Extension Development Host still running?`);
            void vscode.window.showErrorMessage(`Claudelash: port ${port} already in use.`);
        } else {
            log(`Server error: ${err.code ?? 'unknown'}`);
        }
    });

    server.listen(port, '127.0.0.1', () => {
        log(`Listening on http://127.0.0.1:${port}/gesture`);
    });
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    output = vscode.window.createOutputChannel('Claudelash');
    context.subscriptions.push(output);

    log('Claudelash activated (milestone 3 — logging only, no terminal input)');
    await loadGestureMap(context);

    const port = vscode.workspace.getConfiguration('claudelash').get<number>('port', 9247);
    startServer(port);

    context.subscriptions.push(
        vscode.commands.registerCommand('claudelash.showLog', () => output.show(true)),
        vscode.commands.registerCommand('claudelash.reloadGestureMap', async () => {
            await loadGestureMap(context);
            output.show(true);
        }),
        // Milestone 4 needs to find the Claude Code terminal by name. This
        // command exists so you can see exactly what those names are.
        vscode.commands.registerCommand('claudelash.listTerminals', () => {
            const configured = vscode.workspace
                .getConfiguration('claudelash')
                .get<string>('terminalName', 'Claude Code');
            log(`Open terminals (claudelash.terminalName is "${configured}"):`);
            if (vscode.window.terminals.length === 0) {
                log('  (none open)');
            }
            for (const t of vscode.window.terminals) {
                const match = t.name === configured ? '  <-- matches' : '';
                log(`  "${t.name}"${match}`);
            }
            output.show(true);
        }),
    );

    output.show(true);
}

export function deactivate(): void {
    server?.close();
    server = undefined;
}
