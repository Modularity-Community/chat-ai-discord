import * as tf from "@tensorflow/tfjs";
import * as tfn from "@tensorflow/tfjs-node";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import readline from "readline";
import { LabelEncoder } from "danfojs-node";
const { Tokenizer } = require("tf_node_tokenizer");
import * as use from "@tensorflow-models/universal-sentence-encoder";
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

tf.setBackend("wasm").then(async () => {
  console.log("hello tf");
  let modelJson = "./assets/model.json";
  const handler = tfn.io.fileSystem(modelJson);
  const model = await tf.loadLayersModel(handler);
  model.summary();

  let maxLen = 20;
  let data = getData();

  const lblEncoder = labelEncoder(data);
  const t = tokenize(data.trainingSentences);
  //console.log(`Heroes : ${t.textsToSequences(data.trainingSentences)}`);
  const r = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  r.question("User\n", async (inp) => {
    inp.toLowerCase();
    const c = giveTheSameLengthOfArray(t.textsToSequences([inp]), maxLen);
    const input = tf.tensor(c, [1, maxLen]);
    const result = model.predict(input);
    const s = (result as tf.Tensor<tf.Rank>).dataSync();
    const max = Array.from(s).indexOf(Math.max(...s));
    const tag = lblEncoder.inverseTransform([max]);
    for (let i of dataJson["intents"]) {
      if (i["tag"] == tag) {
        console.log(`Bot: ${i["responses"]}`);
      }
    }
    r.close();
  });
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

function giveTheSameLengthOfArray(array: number[][], maxLen: number) {
  const data = array.map((e) => {
    const rowLen = e.length;
    if (rowLen > maxLen) {
      return e.slice(rowLen - maxLen, rowLen);
    }
    if (rowLen < maxLen) {
      return Array(maxLen - rowLen)
        .fill(0)
        .concat(e);
    }
    return e;
  });
  return data;
}

function labelEncoder(j: ModelData): LabelEncoder {
  let encode = new LabelEncoder();
  encode.fit(j.trainingLabels);
  encode.transform(j.trainingLabels);
  return encode;
}

function tokenize(trainingSentences: string[]): any {
  let vocabSize = 1000;
  let oovToken = "<OOV>";
  const tokenizer = new Tokenizer({
    num_words: vocabSize,
    oov_token: oovToken,
  });
  tokenizer.fitOnTexts(trainingSentences);
  return tokenizer;
  //console.log(`Hello ${tokenizer.textsToSequences(trainingSentences)}`);
}
