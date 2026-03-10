class MNISTApp {

constructor() {
    this.dataLoader = new MNISTDataLoader();

    this.modelMax = null;
    this.modelAvg = null;

    this.trainData = null;
    this.testData = null;

    this.isTraining = false;

    this.initializeUI();
}

initializeUI() {

    document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
    document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
    document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
    document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
    document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
    document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
    document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
    document.getElementById('toggleVisorBtn').addEventListener('click', () => tfvis.visor().toggle());

}

async onLoadData() {

    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];

    if (!trainFile || !testFile) {
        this.showError("Select both CSV files");
        return;
    }

    this.showStatus("Loading train data...");
    this.trainData = await this.dataLoader.loadTrainFromFiles(trainFile);

    this.showStatus("Loading test data...");
    this.testData = await this.dataLoader.loadTestFromFiles(testFile);

    this.updateDataStatus(this.trainData.count, this.testData.count);

    this.showStatus("Data loaded successfully");
}

createAutoencoder(poolType) {

    const input = tf.input({shape:[28,28,1]});

    let x = tf.layers.conv2d({
        filters:32,
        kernelSize:3,
        activation:'relu',
        padding:'same'
    }).apply(input);

    if(poolType === "max")
        x = tf.layers.maxPooling2d({poolSize:2}).apply(x);
    else
        x = tf.layers.averagePooling2d({poolSize:2}).apply(x);

    x = tf.layers.conv2d({
        filters:16,
        kernelSize:3,
        activation:'relu',
        padding:'same'
    }).apply(x);

    if(poolType === "max")
        x = tf.layers.maxPooling2d({poolSize:2}).apply(x);
    else
        x = tf.layers.averagePooling2d({poolSize:2}).apply(x);

    x = tf.layers.conv2dTranspose({
        filters:16,
        kernelSize:3,
        strides:2,
        padding:'same',
        activation:'relu'
    }).apply(x);

    x = tf.layers.conv2dTranspose({
        filters:32,
        kernelSize:3,
        strides:2,
        padding:'same',
        activation:'relu'
    }).apply(x);

    const output = tf.layers.conv2d({
        filters:1,
        kernelSize:3,
        activation:'sigmoid',
        padding:'same'
    }).apply(x);

    const model = tf.model({inputs:input, outputs:output});

    model.compile({
        optimizer: tf.train.adam(0.0005),
        loss:'meanSquaredError'
    });

    return model;
}

async onTrain() {

    if (!this.trainData) {
        this.showError("Load data first");
        return;
    }

    if (this.isTraining) return;

    this.isTraining = true;

    this.showStatus("Preparing training...");

    const {trainXs, valXs} = this.dataLoader.splitTrainVal(
        this.trainData.xs,
        this.trainData.ys,
        0.1
    );
    const trainSubset = trainXs.slice([0,0,0,0],[20000,28,28,1]);

    const noisyTrain = addNoise(trainSubset);

    const valSubset = valXs.slice([0,0,0,0],[4000,28,28,1]);
    const noisyVal = addNoise(valSubset);

    this.modelMax = this.createAutoencoder("max");
    this.modelAvg = this.createAutoencoder("avg");

    this.showStatus("Training MAX pooling model...");

    await this.modelMax.fit(noisyTrain, trainSubset,{
        epochs:3,
        batchSize:256,
        validationData:[noisyVal,valSubset],
        callbacks: tfvis.show.fitCallbacks(
            {name:'MaxPool Training'},
            ['loss','val_loss']
        )
    });

    this.showStatus("Training AVG pooling model...");

    await this.modelAvg.fit(noisyTrain, trainSubset,{
        epochs:3,
        batchSize:256,
        validationData:[noisyVal,valSubset],
        callbacks: tfvis.show.fitCallbacks(
            {name:'AvgPool Training'},
            ['loss','val_loss']
        )
    });

    trainXs.dispose();
    valXs.dispose();
    noisyTrain.dispose();
    noisyVal.dispose();

    this.showStatus("Training finished");

    this.updateModelInfo();

    this.isTraining = false;
}

async onEvaluate() {

    if(!this.modelMax || !this.modelAvg || !this.testData){
        this.showError("Train model first");
        return;
    }

    const noisy = addNoise(this.testData.xs);

    const lossMax = this.modelMax.evaluate(noisy,this.testData.xs);
    const lossAvg = this.modelAvg.evaluate(noisy,this.testData.xs);

    const valMax = (await lossMax.data())[0];
    const valAvg = (await lossAvg.data())[0];

    this.showStatus(`Reconstruction Loss MAX: ${valMax.toFixed(5)}`);
    this.showStatus(`Reconstruction Loss AVG: ${valAvg.toFixed(5)}`);

    lossMax.dispose();
    lossAvg.dispose();
    noisy.dispose();
}

async onTestFive(){

    if(!this.modelMax || !this.modelAvg || !this.testData){
        this.showError("Train model first");
        return;
    }

    const container = document.getElementById('previewContainer');
    container.innerHTML="";

    const {batchXs} = this.dataLoader.getRandomTestBatch(
        this.testData.xs,
        this.testData.ys,
        5
    );

    const noisy = addNoise(batchXs);

    const denoiseMax = this.modelMax.predict(noisy);
    const denoiseAvg = this.modelAvg.predict(noisy);

    for(let i=0;i<5;i++){

        const row=document.createElement("div");
        row.style.display="flex";
        row.style.gap="20px";

        const c1=document.createElement("canvas");
        const c2=document.createElement("canvas");
        const c3=document.createElement("canvas");
        const c4=document.createElement("canvas");

        const orig = batchXs.slice([i,0,0,0],[1,28,28,1]).squeeze();
        const noisyImg = noisy.slice([i,0,0,0],[1,28,28,1]).squeeze();
        const maxImg = denoiseMax.slice([i,0,0,0],[1,28,28,1]).squeeze();
        const avgImg = denoiseAvg.slice([i,0,0,0],[1,28,28,1]).squeeze();

        this.dataLoader.draw28x28ToCanvas(orig,c1,4);
        this.dataLoader.draw28x28ToCanvas(noisyImg,c2,4);
        this.dataLoader.draw28x28ToCanvas(maxImg,c3,4);
        this.dataLoader.draw28x28ToCanvas(avgImg,c4,4);

        orig.dispose();
        noisyImg.dispose();
        maxImg.dispose();
        avgImg.dispose();

        row.appendChild(c1);
        row.appendChild(c2);
        row.appendChild(c3);
        row.appendChild(c4);

        container.appendChild(row);
    }


    batchXs.dispose();
    noisy.dispose();
    denoiseMax.dispose();
    denoiseAvg.dispose();
}

async onSaveDownload(){

    if(!this.modelMax){
        this.showError("Train model first");
        return;
    }

    await this.modelMax.save('downloads://mnist-denoiser-max');
    await this.modelAvg.save('downloads://mnist-denoiser-avg');

    this.showStatus("Models saved");
}

async onLoadFromFiles(){

    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const weightsFile = document.getElementById('modelWeightsFile').files[0];

    if(!jsonFile || !weightsFile){
        this.showError("Select model files");
        return;
    }

    this.modelMax = await tf.loadLayersModel(
        tf.io.browserFiles([jsonFile,weightsFile])
    );

    this.showStatus("Model loaded");
}

onReset(){

    if(this.modelMax) this.modelMax.dispose();
    if(this.modelAvg) this.modelAvg.dispose();

    this.modelMax=null;
    this.modelAvg=null;

    this.dataLoader.dispose();

    this.trainData=null;
    this.testData=null;

    document.getElementById('previewContainer').innerHTML="";

    this.updateModelInfo();
}

updateDataStatus(trainCount,testCount){

    const el=document.getElementById("dataStatus");

    el.innerHTML=`
    <h3>Data Status</h3>
    <p>Train samples: ${trainCount}</p>
    <p>Test samples: ${testCount}</p>
    `;
}

updateModelInfo(){

    const el=document.getElementById("modelInfo");

    if(!this.modelMax){
        el.innerHTML="<h3>Model Info</h3><p>No model</p>";
        return;
    }

    el.innerHTML=`
    <h3>Model Info</h3>
    <p>Autoencoder models loaded</p>
    `;
}

showStatus(msg){

    const logs=document.getElementById("trainingLogs");

    const entry=document.createElement("div");

    entry.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;

    logs.appendChild(entry);
}

showError(msg){

    this.showStatus("ERROR: "+msg);
    console.error(msg);
}

}

function addNoise(images, noiseFactor=0.05){

return tf.tidy(()=>{
    const noise=tf.randomNormal(images.shape,0,1);
    const noisy=images.add(noise.mul(noiseFactor));
    return noisy.clipByValue(0,1);
});

}

document.addEventListener("DOMContentLoaded", async ()=>{
await tf.setBackend('cpu')
new MNISTApp();

});
