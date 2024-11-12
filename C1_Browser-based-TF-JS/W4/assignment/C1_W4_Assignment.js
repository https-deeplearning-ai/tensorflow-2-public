let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples = 0, paperSamples = 0, scissorsSamples = 0, spockSamples = 0, lizardSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
  mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);

  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({units: 256, activation: 'relu'}),
      tf.layers.batchNormalization(),
      tf.layers.dropout({rate: 0.3}), // Dropout layer
      tf.layers.dense({units: 128, activation: 'relu'}),
      tf.layers.batchNormalization(),
      tf.layers.dropout({rate: 0.3}), // Dropout layer
      tf.layers.dense({units: 5, activation: 'softmax'}) // Output layer
    ]
  });

  const optimizer = tf.train.adam(0.0001);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

let loss = 0;
  await model.fit(dataset.xs, dataset.ys, {
    epochs: 20, // Meningkatkan jumlah epochs untuk pelatihan yang lebih baik
    batchSize: 32, // Menambahkan batch size
    validationSplit: 0.2, // Menambahkan validation split untuk generalisasi lebih baik
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(5)}, accuracy = ${logs.acc.toFixed(5)}`);
      }
    }
  });
}

function handleButton(elem) {
  switch (elem.id) {
    case "0":
      rockSamples++;
      document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
      break;
    case "2":
      scissorsSamples++;
      document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
      break;
    case "3":
      spockSamples++;
      document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
      break;
    case "4":
      lizardSamples++;
      document.getElementById("lizardsamples").innerText = "Lizard samples:" + lizardSamples;
      break;
  }

  const label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    let predictionText = "";

    switch (classId) {
      case 0:
        predictionText = "I see Rock";
        break;
      case 1:
        predictionText = "I see Paper";
        break;
      case 2:
        predictionText = "I see Scissors";
        break;
      case 3:
        predictionText = "I see Spock";
        break;
      case 4:
        predictionText = "I see Lizard";
        break;
    }
    document.getElementById("prediction").innerText = predictionText;

    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function doTraining() {
  train().then(() => alert("Training Done!"));
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
}

function saveModel() {
  model.save('downloads://my_model');
}

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();
