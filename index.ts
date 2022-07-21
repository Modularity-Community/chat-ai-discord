import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import readline from "readline";
import { LabelEncoder } from "danfojs-node";
const { Tokenizer } = require("tf_node_tokenizer");
import dataJson from "./assets/data.json";
import dotenv from "dotenv";
dotenv.config();

setWasmPaths({
  "tfjs-backend-wasm.wasm": process.env.WASM,
  "tfjs-backend-wasm-simd.wasm": process.env.WASM_SIMD,
  "tfjs-backend-wasm-threaded-simd.wasm": process.env.WASM_THREADED,
});
interface ModelData {
  trainingLabels: string[];
  trainingSentences: string[];
  labels: string[];
  response: string[][];
  num_classes: number;
}

tf.setBackend("wasm").then(() => {
  console.log("hello tf");
  let data = getData();
  labelEncoder(data);
  tokenize(data);
});

function getData(): ModelData {
  const m: ModelData = {
    trainingLabels: [],
    trainingSentences: [],
    labels: [],
    response: [],
    num_classes: 0,
  };
  for (let intent of dataJson["intents"]) {
    for (let pattern of intent["patterns"]) {
      m.trainingSentences.push(pattern);
      m.trainingLabels.push(intent["tag"]);
    }
    m.response.push(intent["responses"]);
    if (!m.labels.includes(intent["tag"])) {
      m.labels.push(intent["tag"]);
    }
  }
  m.num_classes = m.labels.length;
  return m;
}

function labelEncoder(j: ModelData): LabelEncoder {
  let encode = new LabelEncoder();
  encode.fit(j.trainingLabels);
  encode.transform(j.trainingLabels);
  return encode;
}

function tokenize(m: ModelData) {
  let vocabSize = 1000;
  let emmbeddingDim = 16;
  let maxLen = 20;
  let oovToken = "<OOV>";
  const tokenizer = new Tokenizer({
    num_words: vocabSize,
    oov_token: oovToken,
  });
  tokenizer.fitOnTexts(m.trainingSentences);
  const sequences = tokenizer.textsToSequences(m.trainingSentences);
  console.log(sequences);
}

function main() {
  const r = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  r.question("who are you?\n", (name) => {
    console.log(`hey there ${name}`);
    r.close();
  });
}
