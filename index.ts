import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import dotenv from "dotenv";
dotenv.config();
setWasmPaths({
  "tfjs-backend-wasm.wasm": process.env.WASM,
  "tfjs-backend-wasm-simd.wasm": process.env.WASM_SIMD,
  "tfjs-backend-wasm-threaded-simd.wasm": process.env.WASM_THREADED,
});
tf.setBackend("wasm").then(() => {
  console.log("hello tf");
});
