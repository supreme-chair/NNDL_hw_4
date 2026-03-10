class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.denoiserModel = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        // Отключаем WebGL глобально
        this.disableWebGL();
    }

    async disableWebGL() {
        // Принудительно используем CPU
        await tf.setBackend('cpu');
        console.log('CPU backend enabled');
        
        // Отключаем WebGL в настройках
        tf.env().set('WEBGL_VERSION', 0);
        tf.env().set('IS_BROWSER', true);
        tf.env().set('IS_NODE', false);
        
        this.showStatus('WebGL disabled, using CPU');
        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('trainDenoiserBtn').addEventListener('click', () => this.onTrainDenoiser());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('testDenoiseBtn').addEventListener('click', () => this.onTestDenoise());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('saveDenoiserBtn').addEventListener('click', () => this.onSaveDenoiser());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('loadDenoiserBtn').addEventListener('click', () => this.onLoadDenoiserFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
    }

    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                this.showError('Please select both train and test CSV files');
                return;
            }

            this.showStatus('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.showStatus('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.trainData = trainData;
            this.testData = testData;

            // Проверяем данные
            const sample = await trainData.xs.slice([0, 0, 0, 0], [1, 28, 28, 1]).data();
            this.showStatus(`Sample pixel: ${sample[0].toFixed(3)}`);

            this.updateDataStatus(trainData.count, testData.count);
            this.showStatus('Data loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    async onTrain() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.showStatus('Starting training...');
            
            // Используем ОЧЕНЬ мало данных для CPU
            const numSamples = Math.min(500, this.trainData.xs.shape[0]);
            
            const xs = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
            const ys = this.trainData.ys.slice([0, 0], [numSamples, 10]);
            
            const splitIndex = Math.floor(numSamples * 0.8);
            
            const trainXs = xs.slice([0, 0, 0, 0], [splitIndex, 28, 28, 1]);
            const trainYs = ys.slice([0, 0], [splitIndex, 10]);
            
            const valXs = xs.slice([splitIndex, 0, 0, 0], [numSamples - splitIndex, 28, 28, 1]);
            const valYs = ys.slice([splitIndex, 0], [numSamples - splitIndex, 10]);

            if (!this.model) {
                this.model = this.createClassifier();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            
            // Маленький batch size для CPU
            await this.model.fit(trainXs, trainYs, {
                epochs: 3,
                batchSize: 16,
                validationData: [valXs, valYs],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`);
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            this.showStatus(`Training completed in ${duration.toFixed(1)}s`);
            
            tf.dispose([xs, ys, trainXs, trainYs, valXs, valYs]);
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    createClassifier() {
        const model = tf.sequential();
        
        // Минимальная архитектура для CPU
        model.add(tf.layers.flatten({
            inputShape: [28, 28, 1]
        }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async onTrainDenoiser() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.showStatus('Starting denoiser training...');
            
            const numSamples = Math.min(200, this.trainData.xs.shape[0]);
            
            const cleanImages = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
            
            const noise = tf.randomNormal([numSamples, 28, 28, 1], 0, 0.2);
            const noisyImages = tf.tidy(() => {
                return cleanImages.add(noise).clipByValue(0, 1);
            });
            
            const cleanFlat = cleanImages.reshape([numSamples, 784]);
            const noisyFlat = noisyImages.reshape([numSamples, 784]);
            
            const splitIndex = Math.floor(numSamples * 0.8);
            
            const trainNoisy = noisyFlat.slice([0, 0], [splitIndex, 784]);
            const trainClean = cleanFlat.slice([0, 0], [splitIndex, 784]);
            
            const valNoisy = noisyFlat.slice([splitIndex, 0], [numSamples - splitIndex, 784]);
            const valClean = cleanFlat.slice([splitIndex, 0], [numSamples - splitIndex, 784]);

            if (!this.denoiserModel) {
                this.denoiserModel = this.createDenoiser();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            
            await this.denoiserModel.fit(trainNoisy, trainClean, {
                epochs: 3,
                batchSize: 16,
                validationData: [valNoisy, valClean],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            this.showStatus(`Denoiser training completed in ${duration.toFixed(1)}s`);
            
            tf.dispose([cleanImages, noise, noisyImages, cleanFlat, noisyFlat, 
                       trainNoisy, trainClean, valNoisy, valClean]);
            
        } catch (error) {
            this.showError(`Denoiser training failed: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    createDenoiser() {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            inputShape: [784],
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dense({
            units: 784,
            activation: 'sigmoid'
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    async onEvaluate() {
        if (!this.model || !this.testData) {
            this.showError('Please train a classifier first');
            return;
        }

        try {
            this.showStatus('Evaluating...');
            
            const numTest = Math.min(200, this.testData.xs.shape[0]);
            
            const testXs = this.testData.xs.slice([0, 0, 0, 0], [numTest, 28, 28, 1]);
            const testYs = this.testData.ys.slice([0, 0], [numTest, 10]);
            
            const result = this.model.evaluate(testXs, testYs, { batchSize: 16 });
            const acc = result[1].dataSync()[0];
            
            this.showStatus(`Test accuracy: ${(acc * 100).toFixed(2)}%`);
            
            tf.dispose([testXs, testYs, result[0], result[1]]);
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    async onTestFive() {
        if (!this.model || !this.testData) {
            this.showError('Please train a classifier first');
            return;
        }

        try {
            const indices = [];
            for (let i = 0; i < 5; i++) {
                indices.push(Math.floor(Math.random() * this.testData.xs.shape[0]));
            }
            
            const testXs = tf.gather(this.testData.xs, indices);
            const testYs = tf.gather(this.testData.ys, indices);
            
            const predictions = this.model.predict(testXs);
            const predLabels = predictions.argMax(-1);
            const trueLabels = testYs.argMax(-1);
            
            const predArray = await predLabels.array();
            const trueArray = await trueLabels.array();
            
            this.renderImages(testXs, predArray, trueArray);
            
            tf.dispose([testXs, testYs, predictions, predLabels, trueLabels]);
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    renderImages(images, predictions, trueLabels) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '<h3>Results</h3>';
        
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.flexWrap = 'wrap';
        row.style.justifyContent = 'center';
        
        for (let i = 0; i < 5; i++) {
            const canvas = document.createElement('canvas');
            canvas.width = 56;
            canvas.height = 56;
            canvas.style.margin = '5px';
            canvas.style.border = '1px solid #ccc';
            
            const imgTensor = images.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28]);
            const imgData = imgTensor.mul(255).dataSync();
            
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(28, 28);
            
            for (let j = 0; j < 784; j++) {
                const val = imgData[j];
                imageData.data[j * 4] = val;
                imageData.data[j * 4 + 1] = val;
                imageData.data[j * 4 + 2] = val;
                imageData.data[j * 4 + 3] = 255;
            }
            
            ctx.putImageData(imageData, 0, 0);
            
            const div = document.createElement('div');
            div.style.display = 'inline-block';
            div.style.textAlign = 'center';
            div.appendChild(canvas);
            
            const label = document.createElement('div');
            const isCorrect = predictions[i] === trueLabels[i];
            label.style.color = isCorrect ? 'green' : 'red';
            label.textContent = `${predictions[i]}/${trueLabels[i]}`;
            div.appendChild(label);
            
            row.appendChild(div);
            imgTensor.dispose();
        }
        
        container.appendChild(row);
    }

    async onTestDenoise() {
        if (!this.denoiserModel || !this.testData) {
            this.showError('Please train a denoiser first');
            return;
        }

        try {
            const indices = [];
            for (let i = 0; i < 3; i++) {
                indices.push(Math.floor(Math.random() * this.testData.xs.shape[0]));
            }
            
            const original = tf.gather(this.testData.xs, indices);
            
            const noise = tf.randomNormal([3, 28, 28, 1], 0, 0.3);
            const noisy = original.add(noise).clipByValue(0, 1);
            
            const noisyFlat = noisy.reshape([3, 784]);
            const denoisedFlat = this.denoiserModel.predict(noisyFlat);
            const denoised = denoisedFlat.reshape([3, 28, 28, 1]);
            
            const container = document.getElementById('previewContainer');
            container.innerHTML = '<h3>Denoising</h3>';
            
            for (let i = 0; i < 3; i++) {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.justifyContent = 'center';
                row.style.margin = '10px';
                
                const noisyTensor = noisy.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28]);
                const denoisedTensor = denoised.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28]);
                const originalTensor = original.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28]);
                
                row.appendChild(this.tensorToCanvas(noisyTensor, 'Noisy'));
                row.appendChild(this.tensorToCanvas(denoisedTensor, 'Denoised'));
                row.appendChild(this.tensorToCanvas(originalTensor, 'Original'));
                
                container.appendChild(row);
                
                tf.dispose([noisyTensor, denoisedTensor, originalTensor]);
            }
            
            tf.dispose([original, noise, noisy, noisyFlat, denoisedFlat, denoised]);
            
        } catch (error) {
            this.showError(`Denoise test failed: ${error.message}`);
        }
    }

    tensorToCanvas(tensor, label) {
        const div = document.createElement('div');
        div.style.textAlign = 'center';
        div.style.margin = '5px';
        
        const canvas = document.createElement('canvas');
        canvas.width = 56;
        canvas.height = 56;
        
        const data = tensor.mul(255).dataSync();
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);
        
        for (let i = 0; i < 784; i++) {
            const val = data[i];
            imageData.data[i * 4] = val;
            imageData.data[i * 4 + 1] = val;
            imageData.data[i * 4 + 2] = val;
            imageData.data[i * 4 + 3] = 255;
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        div.appendChild(canvas);
        
        const labelDiv = document.createElement('div');
        labelDiv.textContent = label;
        div.appendChild(labelDiv);
        
        return div;
    }

    async onSaveDownload() {
        if (!this.model) return;
        try {
            await this.model.save('downloads://mnist-classifier');
            this.showStatus('Classifier saved');
        } catch (error) {
            this.showError('Save failed');
        }
    }

    async onSaveDenoiser() {
        if (!this.denoiserModel) return;
        try {
            await this.denoiserModel.save('downloads://mnist-denoiser');
            this.showStatus('Denoiser saved');
        } catch (error) {
            this.showError('Save failed');
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) return;
        
        try {
            if (this.model) this.model.dispose();
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            this.showStatus('Classifier loaded');
            this.updateModelInfo();
        } catch (error) {
            this.showError('Load failed');
        }
    }

    async onLoadDenoiserFromFiles() {
        const jsonFile = document.getElementById('denoiserJsonFile').files[0];
        const weightsFile = document.getElementById('denoiserWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) return;
        
        try {
            if (this.denoiserModel) this.denoiserModel.dispose();
            this.denoiserModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            this.showStatus('Denoiser loaded');
            this.updateModelInfo();
        } catch (error) {
            this.showError('Load failed');
        }
    }

    onReset() {
        if (this.model) this.model.dispose();
        if (this.denoiserModel) this.denoiserModel.dispose();
        this.model = null;
        this.denoiserModel = null;
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        document.getElementById('previewContainer').innerHTML = '';
        this.showStatus('Reset completed');
    }

    toggleVisor() {
        try {
            if (tfvis.visor().isOpen()) {
                tfvis.visor().close();
            } else {
                tfvis.visor().open();
            }
        } catch (error) {
            this.showError('Visor toggle failed');
        }
    }

    updateDataStatus(trainCount, testCount) {
        document.getElementById('dataStatus').innerHTML = `
            <h3>Data Status</h3>
            <p>Train: ${trainCount}</p>
            <p>Test: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const info = document.getElementById('modelInfo');
        info.innerHTML = `
            <h3>Models</h3>
            <p>Classifier: ${this.model ? '✓' : '✗'}</p>
            <p>Denoiser: ${this.denoiserModel ? '✓' : '✗'}</p>
        `;
    }

    showStatus(msg) {
        const logs = document.getElementById('trainingLogs');
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
    }

    showError(msg) {
        this.showStatus(`ERROR: ${msg}`);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        new MNISTApp();
    }, 100);
});