import { FMnistData } from './fashion-data.js';

var canvas, ctx, saveButton, clearButton;
var pos = { x: 0, y: 0 };
var rawImage;
var model;

function getModel() {
    // Membuat model CNN
    model = tf.sequential();

    // Menambahkan layer konvolusi, pooling, flatten, dan dense
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax' // Output layer untuk 10 kelas
    }));

    // Mengompilasi model
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

async function train(model, data) {
    // Set metrics untuk callback
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];

    // Membuat container untuk callback
    const container = document.getElementById('fitCallbacksContainer');
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 60000; // Ukuran data pelatihan
    const TEST_DATA_SIZE = 10000; // Ukuran data pengujian

    // Mendapatkan batch pelatihan dan mengubah ukuran
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.getTrainData();
        return [d.xs, d.ys]; // Tidak perlu reshape jika sudah dalam bentuk tensor
    });

    // Mendapatkan batch pengujian dan mengubah ukuran
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.getTestData();
        return [d.xs, d.ys]; // Tidak perlu reshape jika sudah dalam bentuk tensor
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

function setPosition(e) {
    pos.x = e.clientX - canvas.offsetLeft; // Menghitung posisi dengan benar
    pos.y = e.clientY - canvas.offsetTop; // Menghitung posisi dengan benar
}

function draw(e) {
    if (e.buttons !== 1) return; // Menggunakan strict equality
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}

function erase() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Menggunakan ukuran kanvas
}

async function save() {
    var raw = tf.browser.fromPixels(rawImage);
    var resized = tf.image.resizeBilinear(raw, [28, 28]);
    var tensor = resized.expandDims(0).toFloat().div(tf.scalar(255)); // Normalisasi

    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync()[0]; // Ambil indeks pertama

    var classNames = ["T-shirt/top", "Trouser", "Pullover",
                      "Dress", "Coat", "Sandal", "Shirt",
                      "Sneaker", "Bag", "Ankle boot"];

    alert(classNames[pIndex]);
}

function init() {
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasimg');
    ctx = canvas.getContext("2d", { willReadFrequently: true }); // Menambahkan opsi ini
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Menggunakan ukuran kanvas
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    saveButton = document.getElementById('sb');
    saveButton.addEventListener("click", save);
    clearButton = document.getElementById('cb');
    clearButton.addEventListener("click", erase);
}

async function run() {
    const data = new FMnistData();
    await data.load();
    model = getModel();
    tfvis.show.modelSummary({ name: 'Model Architecture' }, model);
    await train(model, data);
    await model.save('downloads://my_model');
    init();
    alert("Training is done, try classifying your drawings!");
}

// Menjalankan fungsi run setelah DOM siap
document.addEventListener('DOMContentLoaded', run);
