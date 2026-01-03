//! Web Worker support for background transcoding.

use crate::error::{ErrorCode, WasmError, WasmResult};
use js_sys::{Array, Object, Reflect};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Message types for worker communication.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerMessageType {
    /// Initialize the worker.
    Init = 0,
    /// Start transcoding.
    Start = 1,
    /// Add input data chunk.
    AddData = 2,
    /// Finish input.
    Finish = 3,
    /// Cancel transcoding.
    Cancel = 4,
    /// Progress update.
    Progress = 5,
    /// Output data available.
    Output = 6,
    /// Transcoding complete.
    Complete = 7,
    /// Error occurred.
    Error = 8,
}

/// Worker message structure.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WorkerMessage {
    /// Message type.
    message_type: WorkerMessageType,
    /// Message payload (as JSON string).
    payload: String,
}

#[wasm_bindgen]
impl WorkerMessage {
    /// Create a new worker message.
    #[wasm_bindgen(constructor)]
    pub fn new(message_type: WorkerMessageType, payload: &str) -> Self {
        Self {
            message_type,
            payload: payload.to_string(),
        }
    }

    /// Create an init message.
    pub fn init() -> Self {
        Self::new(WorkerMessageType::Init, "{}")
    }

    /// Create a start message with options.
    #[wasm_bindgen(js_name = start)]
    pub fn start(options_json: &str) -> Self {
        Self::new(WorkerMessageType::Start, options_json)
    }

    /// Create an add data message.
    #[wasm_bindgen(js_name = addData)]
    pub fn add_data() -> Self {
        Self::new(WorkerMessageType::AddData, "{}")
    }

    /// Create a finish message.
    pub fn finish() -> Self {
        Self::new(WorkerMessageType::Finish, "{}")
    }

    /// Create a cancel message.
    pub fn cancel() -> Self {
        Self::new(WorkerMessageType::Cancel, "{}")
    }

    /// Create a progress message.
    pub fn progress(progress: f64, frames: u64) -> Self {
        Self::new(
            WorkerMessageType::Progress,
            &format!(r#"{{"progress":{},"frames":{}}}"#, progress, frames),
        )
    }

    /// Create an output message.
    pub fn output(bytes: u64) -> Self {
        Self::new(
            WorkerMessageType::Output,
            &format!(r#"{{"bytes":{}}}"#, bytes),
        )
    }

    /// Create a complete message.
    pub fn complete(stats_json: &str) -> Self {
        Self::new(WorkerMessageType::Complete, stats_json)
    }

    /// Create an error message.
    pub fn error(message: &str, code: ErrorCode) -> Self {
        Self::new(
            WorkerMessageType::Error,
            &format!(r#"{{"message":"{}","code":{}}}"#, message, code as u32),
        )
    }

    /// Get message type.
    #[wasm_bindgen(getter, js_name = messageType)]
    pub fn message_type(&self) -> WorkerMessageType {
        self.message_type
    }

    /// Get payload.
    #[wasm_bindgen(getter)]
    pub fn payload(&self) -> String {
        self.payload.clone()
    }

    /// Convert to JavaScript object.
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(&obj, &"type".into(), &(self.message_type as u32).into());
        let _ = Reflect::set(&obj, &"payload".into(), &self.payload.clone().into());
        obj
    }

    /// Parse from JavaScript object.
    #[wasm_bindgen(js_name = fromObject)]
    pub fn from_object(obj: &Object) -> WasmResult<WorkerMessage> {
        let type_val = Reflect::get(obj, &"type".into())
            .map_err(|_| WasmError::invalid_input("Missing type field"))?;

        let type_num = type_val
            .as_f64()
            .ok_or_else(|| WasmError::invalid_input("Type must be a number"))?
            as u32;

        let message_type = match type_num {
            0 => WorkerMessageType::Init,
            1 => WorkerMessageType::Start,
            2 => WorkerMessageType::AddData,
            3 => WorkerMessageType::Finish,
            4 => WorkerMessageType::Cancel,
            5 => WorkerMessageType::Progress,
            6 => WorkerMessageType::Output,
            7 => WorkerMessageType::Complete,
            8 => WorkerMessageType::Error,
            _ => return Err(WasmError::invalid_input("Unknown message type")),
        };

        let payload = Reflect::get(obj, &"payload".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| "{}".to_string());

        Ok(Self {
            message_type,
            payload,
        })
    }
}

/// Worker pool for parallel transcoding.
#[wasm_bindgen]
pub struct WorkerPool {
    /// Maximum number of workers.
    max_workers: usize,
    /// Active workers.
    workers: Vec<web_sys::Worker>,
    /// Pending tasks.
    pending_tasks: usize,
    /// Completed tasks.
    completed_tasks: usize,
}

#[wasm_bindgen]
impl WorkerPool {
    /// Create a new worker pool.
    #[wasm_bindgen(constructor)]
    pub fn new(max_workers: usize) -> Self {
        Self {
            max_workers: max_workers.max(1),
            workers: Vec::new(),
            pending_tasks: 0,
            completed_tasks: 0,
        }
    }

    /// Create a worker pool with automatic sizing.
    pub fn auto() -> Self {
        // Try to detect hardware concurrency
        let concurrency = js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
            .ok()
            .and_then(|nav| js_sys::Reflect::get(&nav, &"hardwareConcurrency".into()).ok())
            .and_then(|hc| hc.as_f64())
            .map(|n| n as usize)
            .unwrap_or(4);

        Self::new(concurrency)
    }

    /// Get maximum worker count.
    #[wasm_bindgen(getter, js_name = maxWorkers)]
    pub fn max_workers(&self) -> usize {
        self.max_workers
    }

    /// Get active worker count.
    #[wasm_bindgen(getter, js_name = activeWorkers)]
    pub fn active_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get pending task count.
    #[wasm_bindgen(getter, js_name = pendingTasks)]
    pub fn pending_tasks(&self) -> usize {
        self.pending_tasks
    }

    /// Get completed task count.
    #[wasm_bindgen(getter, js_name = completedTasks)]
    pub fn completed_tasks(&self) -> usize {
        self.completed_tasks
    }

    /// Create a new worker.
    #[wasm_bindgen(js_name = createWorker)]
    pub fn create_worker(&mut self, script_url: &str) -> WasmResult<usize> {
        if self.workers.len() >= self.max_workers {
            return Err(WasmError::new(
                "Worker pool is at maximum capacity",
                ErrorCode::InvalidConfig,
            ));
        }

        let options = web_sys::WorkerOptions::new();
        options.set_type(web_sys::WorkerType::Module);

        let worker = web_sys::Worker::new_with_options(script_url, &options)
            .map_err(|e| WasmError::new(&format!("Failed to create worker: {:?}", e), ErrorCode::BrowserNotSupported))?;

        let index = self.workers.len();
        self.workers.push(worker);

        Ok(index)
    }

    /// Post a message to a worker.
    #[wasm_bindgen(js_name = postMessage)]
    pub fn post_message(&self, worker_index: usize, message: &WorkerMessage) -> WasmResult<()> {
        if worker_index >= self.workers.len() {
            return Err(WasmError::invalid_input("Invalid worker index"));
        }

        let worker = &self.workers[worker_index];
        worker
            .post_message(&message.to_object())
            .map_err(|e| WasmError::new(&format!("Failed to post message: {:?}", e), ErrorCode::IoError))?;

        Ok(())
    }

    /// Post a message with transferable objects.
    #[wasm_bindgen(js_name = postMessageWithTransfer)]
    pub fn post_message_with_transfer(
        &self,
        worker_index: usize,
        message: &WorkerMessage,
        transfer: &Array,
    ) -> WasmResult<()> {
        if worker_index >= self.workers.len() {
            return Err(WasmError::invalid_input("Invalid worker index"));
        }

        let worker = &self.workers[worker_index];
        worker
            .post_message_with_transfer(&message.to_object(), transfer)
            .map_err(|e| WasmError::new(&format!("Failed to post message: {:?}", e), ErrorCode::IoError))?;

        Ok(())
    }

    /// Set message handler for a worker.
    #[wasm_bindgen(js_name = setOnMessage)]
    pub fn set_on_message(&self, worker_index: usize, callback: &js_sys::Function) -> WasmResult<()> {
        if worker_index >= self.workers.len() {
            return Err(WasmError::invalid_input("Invalid worker index"));
        }

        let worker = &self.workers[worker_index];
        worker.set_onmessage(Some(callback));

        Ok(())
    }

    /// Set error handler for a worker.
    #[wasm_bindgen(js_name = setOnError)]
    pub fn set_on_error(&self, worker_index: usize, callback: &js_sys::Function) -> WasmResult<()> {
        if worker_index >= self.workers.len() {
            return Err(WasmError::invalid_input("Invalid worker index"));
        }

        let worker = &self.workers[worker_index];
        worker.set_onerror(Some(callback));

        Ok(())
    }

    /// Terminate a worker.
    #[wasm_bindgen(js_name = terminateWorker)]
    pub fn terminate_worker(&mut self, worker_index: usize) -> WasmResult<()> {
        if worker_index >= self.workers.len() {
            return Err(WasmError::invalid_input("Invalid worker index"));
        }

        let worker = self.workers.remove(worker_index);
        worker.terminate();

        Ok(())
    }

    /// Terminate all workers.
    #[wasm_bindgen(js_name = terminateAll)]
    pub fn terminate_all(&mut self) {
        for worker in self.workers.drain(..) {
            worker.terminate();
        }
        self.pending_tasks = 0;
    }

    /// Get pool statistics.
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(
            &obj,
            &"maxWorkers".into(),
            &JsValue::from_f64(self.max_workers as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"activeWorkers".into(),
            &JsValue::from_f64(self.workers.len() as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"pendingTasks".into(),
            &JsValue::from_f64(self.pending_tasks as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"completedTasks".into(),
            &JsValue::from_f64(self.completed_tasks as f64),
        );
        obj
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.terminate_all();
    }
}

/// Helper function to run transcoding in a worker.
#[wasm_bindgen(js_name = runInWorker)]
pub fn run_in_worker(script_url: &str) -> WasmResult<web_sys::Worker> {
    let options = web_sys::WorkerOptions::new();
    options.set_type(web_sys::WorkerType::Module);

    web_sys::Worker::new_with_options(script_url, &options)
        .map_err(|e| WasmError::new(&format!("Failed to create worker: {:?}", e), ErrorCode::BrowserNotSupported))
}

/// Generate worker script content for embedding.
#[wasm_bindgen(js_name = getWorkerScript)]
pub fn get_worker_script(wasm_url: &str) -> String {
    format!(
        r#"
import init, {{ StreamingTranscoder }} from '{}';

let transcoder = null;

self.onmessage = async (event) => {{
    const {{ type, payload, data }} = event.data;

    try {{
        switch (type) {{
            case 0: // Init
                await init();
                transcoder = new StreamingTranscoder();
                self.postMessage({{ type: 0, payload: '{{}}' }});
                break;

            case 1: // Start
                // Configure transcoder with options from payload
                self.postMessage({{ type: 1, payload: '{{}}' }});
                break;

            case 2: // AddData
                if (transcoder && data) {{
                    const output = transcoder.processChunk(new Uint8Array(data));
                    if (output.length > 0) {{
                        self.postMessage(
                            {{ type: 6, payload: '{{}}' }},
                            [output.buffer]
                        );
                    }}
                    // Send progress
                    self.postMessage({{
                        type: 5,
                        payload: JSON.stringify({{
                            framesProcessed: transcoder.framesProcessed,
                            inputBytes: transcoder.inputBytes,
                            outputBytes: transcoder.outputBytes
                        }})
                    }});
                }}
                break;

            case 3: // Finish
                if (transcoder) {{
                    const output = transcoder.finish();
                    self.postMessage(
                        {{ type: 7, payload: JSON.stringify(transcoder.stats) }},
                        output.length > 0 ? [output.buffer] : undefined
                    );
                }}
                break;

            case 4: // Cancel
                if (transcoder) {{
                    transcoder.reset();
                }}
                break;
        }}
    }} catch (error) {{
        self.postMessage({{
            type: 8,
            payload: JSON.stringify({{ message: error.message, code: 0 }})
        }});
    }}
}};
"#,
        wasm_url
    )
}

/// Create a blob URL for the worker script.
#[wasm_bindgen(js_name = createWorkerBlobUrl)]
pub fn create_worker_blob_url(wasm_url: &str) -> WasmResult<String> {
    let script = get_worker_script(wasm_url);

    let parts = Array::new();
    parts.push(&JsValue::from_str(&script));

    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/javascript");

    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|e| WasmError::new(&format!("Failed to create blob: {:?}", e), ErrorCode::IoError))?;

    web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| WasmError::new(&format!("Failed to create URL: {:?}", e), ErrorCode::IoError))
}
