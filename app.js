class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.denoiserModel = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        // Принудительно используем CPU
        this.initBackend();
        this.initializeUI();
    }

    async initBackend() {
        try {
            // Явно устанавливаем CPU бэкенд
            await tf.setBackend('cpu');
            console.log('Using CPU backend');
            this.showStatus('Using CPU backend for stability');
        } catch (error) {
            console.error('Failed to set backend:', error);
            this.showError('Failed to initialize TensorFlow.js');
        }
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
        
        // Убираем выбор pooling type для простоты
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
            
            // Используем очень маленькое подмножество данных для теста
            const numSamples = Math.min(1000, this.trainData.xs.shape[0]);
            const subsetXs = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
            const subsetYs = this.trainData.ys.slice([0, 0], [numSamples, 10]);
            
            const splitIndex = Math.floor(numSamples * 0.9);
            
            const trainXs = subsetXs.slice([0, 0, 0, 0], [splitIndex, 28, 28, 1]);
            const trainYs = subsetYs.slice([0, 0], [splitIndex, 10]);
            
            const valXs = subsetXs.slice([splitIndex, 0, 0, 0], [numSamples - splitIndex, 28, 28, 1]);
            const valYs = subsetYs.slice([splitIndex, 0], [numSamples - splitIndex, 10]);

            if (!this.model) {
                this.model = this.createSimpleClassifier();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            
            // Обучаем без визуализации для экономии памяти
            await this.model.fit(trainXs, trainYs, {
                epochs: 2,
                batchSize: 32,
                validationData: [valXs, valYs],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`);
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            this.showStatus(`Training completed in ${duration.toFixed(1)}s`);
            
            // Очищаем память
            tf.dispose([subsetXs, subsetYs, trainXs, trainYs, valXs, valYs]);
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    createSimpleClassifier() {
        // Максимально простая модель
        const model = tf.sequential();
        
        model.add(tf.layers.flatten({
            inputShape: [28, 28, 1]
        }));
        
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));
        
        model.compile({
            optimizer: 'adam',
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
            
            // Используем очень маленькое подмножество
            const numSamples = Math.min(500, this.trainData.xs.shape[0]);
            const subsetXs = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
            
            // Создаем зашумленные данные
            const noise = tf.randomNormal([numSamples, 28, 28, 1], 0, 0.2);
            const noisyXs = subsetXs.add(noise).clipByValue(0, 1);
            
            const splitIndex = Math.floor(numSamples * 0.9);
            
            const trainNoisy = noisyXs.slice([0, 0, 0, 0], [splitIndex, 28, 28, 1]);
            const trainClean = subsetXs.slice([0, 0, 0, 0], [splitIndex, 28, 28, 1]);
            
            const valNoisy = noisyXs.slice([splitIndex, 0, 0, 0], [numSamples - splitIndex, 28, 28, 1]);
            const valClean = subsetXs.slice([splitIndex, 0, 0, 0], [numSamples - splitIndex, 28, 28, 1]);

            if (!this.denoiserModel) {
                this.denoiserModel = this.createSimpleDenoiser();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            
            await this.denoiserModel.fit(trainNoisy, trainClean, {
                epochs: 3,
                batchSize: 32,
                validationData: [valNoisy, valClean],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            this.showStatus(`Denoiser training completed in ${duration.toFixed(1)}s`);
            
            // Очищаем память
            tf.dispose([subsetXs, noise, noisyXs, trainNoisy, trainClean, valNoisy, valClean]);
            
        } catch (error) {
            this.showError(`Denoiser training failed: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    createSimpleDenoiser() {
        // Простой автоэнкодер
        const model = tf.sequential();
        
        // Encoder
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [784]
        }));
        
        // Decoder
        model.add(tf.layers.dense({
            units: 784,
            activation: 'sigmoid'
        }));
        
        model.compile({
            optimizer: 'adam',
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
            
            // Используем маленькое подмножество
            const numTest = Math.min(200, this.testData.xs.shape[0]);
            const testXs = this.testData.xs.slice([0, 0, 0, 0], [numTest, 28, 28, 1]);
            const testYs = this.testData.ys.slice([0, 0], [numTest, 10]);
            
            const result = this.model.evaluate(testXs, testYs);
            const loss = result[0].dataSync()[0];
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
            
            const batchXs = tf.gather(this.testData.xs, indices);
            const batchYs = tf.gather(this.testData.ys, indices);
            
            const predictions = this.model.predict(batchXs);
            const predLabels = predictions.argMax(-1);
            const trueLabels = batchYs.argMax(-1);
            
            const predArray = await predLabels.array();
            const trueArray = await trueLabels.array();
            
            this.renderSimplePreview(batchXs, predArray, trueArray);
            
            tf.dispose([batchXs, batchYs, predictions, predLabels, trueLabels]);
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    async onTestDenoise() {
        if (!this.denoiserModel || !this.testData) {
            this.showError('Please train a denoiser first');
            return;
        }

        try {
            this.showStatus('Testing denoiser...');
            
            const indices = [];
            for (let i = 0; i < 3; i++) {
                indices.push(Math.floor(Math.random() * this.testData.xs.shape[0]));
            }
            
            const original = tf.gather(this.testData.xs, indices);
            
            // Добавляем шум
            const noise = tf.randomNormal([3, 28, 28, 1], 0, 0.3);
            const noisy = original.add(noise).clipByValue(0, 1);
            
            // Для денойзера нужно преобразовать в вектор
            const noisyFlat = noisy.reshape([3, 784]);
            const denoisedFlat = this.denoiserModel.predict(noisyFlat);
            const denoised = denoisedFlat.reshape([3, 28, 28, 1]);
            
            this.renderDenoisePreview(noisy, denoised, original);
            
            tf.dispose([original, noise, noisy, noisyFlat, denoisedFlat, denoised]);
            
        } catch (error) {
            this.showError(`Denoise test failed: ${error.message}`);
        }
    }

    renderSimplePreview(images, predictions, trueLabels) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        for (let i = 0; i < 5; i++) {
            const canvas = document.createElement('canvas');
            canvas.width = 56;
            canvas.height = 56;
            canvas.style.margin = '5px';
            
            const tensor = images.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28]);
            const data = tensor.mul(255).dataSync();
            
            const ctx = canvas.getContext('2d');
            const imgData = ctx.createImageData(28, 28);
            
            for (let j = 0; j < 784; j++) {
                imgData.data[j * 4] = data[j];
                imgData.data[j * 4 + 1] = data[j];
                imgData.data[j * 4 + 2] = data[j];
                imgData.data[j * 4 + 3] = 255;
            }
            
            ctx.putImageData(imgData, 0, 0);
            
            const div = document.createElement('div');
            div.style.display = 'inline-block';
            div.style.textAlign = 'center';
            div.appendChild(canvas);
            
            const label = document.createElement('div');
            const isCorrect = predictions[i] === trueLabels[i];
            label.style.color = isCorrect ? 'green' : 'red';
            label.textContent = `P:${predictions[i]} T:${trueLabels[i]}`;
            div.appendChild(label);
            
            container.appendChild(div);
            
            tensor.dispose();
        }
    }

    renderDenoisePreview(noisy, denoised, original) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '<h3>Denoising Results (Left: Noisy, Middle: Denoised, Right: Original)</h3>';
        
        for (let i = 0; i < 3; i++) {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.justifyContent = 'center';
            row.style.marginBottom = '10px';
            
            [noisy, denoised, original].forEach((tensor, idx) => {
                const canvas = document.createElement('canvas');
                canvas.width = 56;
                canvas.height = 56;
                canvas.style.margin = '5px';
                
                const imgTensor = tensor.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28]);
                const data = imgTensor.mul(255).dataSync();
                
                const ctx = canvas.getContext('2d');
                const imgData = ctx.createImageData(28, 28);
                
                for (let j = 0; j < 784; j++) {
                    imgData.data[j * 4] = data[j];
                    imgData.data[j * 4 + 1] = data[j];
                    imgData.data[j * 4 + 2] = data[j];
                    imgData.data[j * 4 + 3] = 255;
                }
                
                ctx.putImageData(imgData, 0, 0);
                row.appendChild(canvas);
                
                imgTensor.dispose();
            });
            
            container.appendChild(row);
        }
    }

    async onSaveDownload() {
        if (!this.model) return;
        try {
            await this.model.save('downloads://mnist-model');
            this.showStatus('Model saved');
        } catch (error) {
            this.showError('Save failed');
        }
    }

    async onSaveDenoiser() {
        if (!this.denoiserModel) return;
        try {
            await this.denoiserModel.save('downloads://denoiser-model');
            this.showStatus('Denoiser saved');
        } catch (error) {
            this.showError('Save failed');
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Select both files');
            return;
        }

        try {
            if (this.model) this.model.dispose();
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            this.showStatus('Model loaded');
            this.updateModelInfo();
        } catch (error) {
            this.showError('Load failed');
        }
    }

    async onLoadDenoiserFromFiles() {
        const jsonFile = document.getElementById('denoiserJsonFile').files[0];
        const weightsFile = document.getElementById('denoiserWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Select both files');
            return;
        }

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
        tfvis.visor().toggle();
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
            <h3>Model Status</h3>
            <p>Classifier: ${this.model ? 'Loaded' : 'Not loaded'}</p>
            <p>Denoiser: ${this.denoiserModel ? 'Loaded' : 'Not loaded'}</p>
        `;
    }

    showStatus(msg) {
        const logs = document.getElementById('trainingLogs');
        logs.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${msg}</div>`;
        logs.scrollTop = logs.scrollHeight;
    }

    showError(msg) {
        this.showStatus(`ERROR: ${msg}`);
    }
}

// Запускаем после загрузки страницы
document.addEventListener('DOMContentLoaded', () => {
    // Небольшая задержка для инициализации
    setTimeout(() => {
        new MNISTApp();
    }, 100);
});
