class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.denoiserModel = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        // Устанавливаем бэкенд CPU для стабильности
        this.setBackend();
        this.initializeUI();
    }

    async setBackend() {
        try {
            // Пробуем использовать WebGL, если не получается - переключаемся на CPU
            await tf.setBackend('webgl');
            console.log('Using WebGL backend');
        } catch (error) {
            await tf.setBackend('cpu');
            console.log('Using CPU backend');
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
        
        document.getElementById('poolingType').addEventListener('change', (e) => {
            this.poolingType = e.target.value;
        });
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
            
            // Уменьшаем размер батча для стабильности
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );

            if (!this.model) {
                this.model = this.createClassifierModel();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            const history = await this.model.fit(trainXs, trainYs, {
                epochs: 3, // Уменьшаем количество эпох для тестирования
                batchSize: 64, // Уменьшаем размер батча
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`);
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            const bestValAcc = Math.max(...history.history.val_acc);
            
            this.showStatus(`Training completed in ${duration.toFixed(1)}s. Best val_acc: ${(bestValAcc * 100).toFixed(2)}%`);
            
            tf.dispose([trainXs, trainYs, valXs, valYs]);
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
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
            
            // Используем меньше данных для тренировки
            const numSamples = Math.min(10000, this.trainData.xs.shape[0]);
            const subsetXs = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
            
            const noisyTrainXs = this.addNoiseToTensor(subsetXs, 0.2);
            
            const splitIndex = Math.floor(numSamples * 0.9);
            
            const trainNoisy = noisyTrainXs.slice([0, 0, 0, 0], [splitIndex, 28, 28, 1]);
            const trainClean = subsetXs.slice([0, 0, 0, 0], [splitIndex, 28, 28, 1]);
            
            const valNoisy = noisyTrainXs.slice([splitIndex, 0, 0, 0], [numSamples - splitIndex, 28, 28, 1]);
            const valClean = subsetXs.slice([splitIndex, 0, 0, 0], [numSamples - splitIndex, 28, 28, 1]);

            if (!this.denoiserModel) {
                this.denoiserModel = this.createDenoiserModel();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            const history = await this.denoiserModel.fit(trainNoisy, trainClean, {
                epochs: 5,
                batchSize: 64,
                validationData: [valNoisy, valClean],
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            this.showStatus(`Denoiser training completed in ${duration.toFixed(1)}s`);
            
            tf.dispose([noisyTrainXs, trainNoisy, trainClean, valNoisy, valClean, subsetXs]);
            
        } catch (error) {
            this.showError(`Denoiser training failed: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    addNoiseToTensor(tensor, noiseFactor = 0.2) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(tensor.shape, 0, noiseFactor);
            const noisy = tensor.add(noise);
            return noisy.clipByValue(0, 1);
        });
    }

    createDenoiserModel() {
        const model = tf.sequential();
        
        // Упрощенная архитектура для стабильности
        model.add(tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        
        model.add(tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        model.add(tf.layers.upSampling2d({ size: 2 }));
        
        model.add(tf.layers.conv2d({
            filters: 1,
            kernelSize: 3,
            activation: 'sigmoid',
            padding: 'same'
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    createClassifierModel() {
        const model = tf.sequential();
        
        // Упрощенная архитектура классификатора
        model.add(tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: 'relu',
            inputShape: [28, 28, 1]
        }));
        
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async onEvaluate() {
        if (!this.model) {
            this.showError('No model available. Please train or load a model first.');
            return;
        }

        if (!this.testData) {
            this.showError('No test data available');
            return;
        }

        try {
            this.showStatus('Evaluating model...');
            
            // Используем подмножество тестовых данных
            const numTest = Math.min(1000, this.testData.xs.shape[0]);
            const testXs = this.testData.xs.slice([0, 0, 0, 0], [numTest, 28, 28, 1]);
            const testYs = this.testData.ys.slice([0, 0], [numTest, 10]);
            
            const result = this.model.evaluate(testXs, testYs, { batchSize: 64 });
            const loss = Array.isArray(result) ? result[0].dataSync()[0] : result.dataSync()[0];
            const accuracy = Array.isArray(result) ? result[1].dataSync()[0] : 0;
            
            this.showStatus(`Test loss: ${loss.toFixed(4)}, accuracy: ${(accuracy * 100).toFixed(2)}%`);
            
            tf.dispose([testXs, testYs]);
            if (Array.isArray(result)) {
                result.forEach(r => r.dispose());
            } else {
                result.dispose();
            }
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    async onTestFive() {
        if (!this.model || !this.testData) {
            this.showError('Please load both model and test data first');
            return;
        }

        try {
            const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            
            const predictions = this.model.predict(batchXs);
            const predictedLabels = predictions.argMax(-1);
            const trueLabels = batchYs.argMax(-1);
            
            const predArray = await predictedLabels.array();
            const trueArray = await trueLabels.array();
            
            this.renderPreview(batchXs, predArray, trueArray, indices);
            
            tf.dispose([predictions, predictedLabels, trueLabels, batchXs, batchYs]);
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    async onTestDenoise() {
        if (!this.denoiserModel || !this.testData) {
            this.showError('Please train or load a denoiser model first');
            return;
        }

        try {
            this.showStatus('Testing denoiser...');
            
            const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 3 // Показываем 3 изображения для наглядности
            );
            
            const noisyBatch = this.addNoiseToTensor(batchXs, 0.3);
            const denoisedBatch = this.denoiserModel.predict(noisyBatch);
            
            this.renderDenoisePreview(noisyBatch, denoisedBatch, batchXs, indices);
            
            tf.dispose([noisyBatch, denoisedBatch, batchXs, batchYs]);
            
        } catch (error) {
            this.showError(`Denoise test failed: ${error.message}`);
        }
    }

    renderDenoisePreview(noisy, denoised, original, indices) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '<h3>Denoising Results</h3>';
        
        const processTensor = (tensor) => {
            return tf.tidy(() => {
                return tensor.squeeze().mul(255).clipByValue(0, 255);
            });
        };
        
        for (let i = 0; i < 3; i++) {
            const row = document.createElement('div');
            row.className = 'preview-row';
            row.style.display = 'flex';
            row.style.justifyContent = 'center';
            row.style.gap = '20px';
            row.style.marginBottom = '20px';
            
            // Noisy
            const noisyItem = document.createElement('div');
            noisyItem.className = 'preview-item';
            const noisyCanvas = document.createElement('canvas');
            noisyCanvas.width = 84;
            noisyCanvas.height = 84;
            const noisyTensor = processTensor(noisy.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.drawTensorToCanvas(noisyTensor, noisyCanvas);
            noisyItem.appendChild(noisyCanvas);
            noisyItem.appendChild(document.createTextNode('Noisy'));
            
            // Denoised
            const denoisedItem = document.createElement('div');
            denoisedItem.className = 'preview-item';
            const denoisedCanvas = document.createElement('canvas');
            denoisedCanvas.width = 84;
            denoisedCanvas.height = 84;
            const denoisedTensor = processTensor(denoised.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.drawTensorToCanvas(denoisedTensor, denoisedCanvas);
            denoisedItem.appendChild(denoisedCanvas);
            denoisedItem.appendChild(document.createTextNode('Denoised'));
            
            // Original
            const originalItem = document.createElement('div');
            originalItem.className = 'preview-item';
            const originalCanvas = document.createElement('canvas');
            originalCanvas.width = 84;
            originalCanvas.height = 84;
            const originalTensor = processTensor(original.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.drawTensorToCanvas(originalTensor, originalCanvas);
            originalItem.appendChild(originalCanvas);
            originalItem.appendChild(document.createTextNode('Original'));
            
            row.appendChild(noisyItem);
            row.appendChild(denoisedItem);
            row.appendChild(originalItem);
            container.appendChild(row);
            
            tf.dispose([noisyTensor, denoisedTensor, originalTensor]);
        }
    }

    drawTensorToCanvas(tensor, canvas) {
        return tf.tidy(() => {
            const ctx = canvas.getContext('2d');
            const data = tensor.dataSync();
            const imgData = ctx.createImageData(28, 28);
            
            for (let i = 0; i < 28 * 28; i++) {
                const val = data[i];
                imgData.data[i * 4] = val;
                imgData.data[i * 4 + 1] = val;
                imgData.data[i * 4 + 2] = val;
                imgData.data[i * 4 + 3] = 255;
            }
            
            ctx.putImageData(imgData, 0, 0);
            ctx.imageSmoothingEnabled = false;
        });
    }

    async onSaveDownload() {
        if (!this.model) {
            this.showError('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-classifier');
            this.showStatus('Classifier model saved successfully!');
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
        }
    }

    async onSaveDenoiser() {
        if (!this.denoiserModel) {
            this.showError('No denoiser model to save');
            return;
        }

        try {
            await this.denoiserModel.save('downloads://mnist-denoiser');
            this.showStatus('Denoiser model saved successfully!');
        } catch (error) {
            this.showError(`Failed to save denoiser model: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading classifier model...');
            
            if (this.model) {
                this.model.dispose();
            }
            
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            this.showStatus('Classifier model loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    async onLoadDenoiserFromFiles() {
        const jsonFile = document.getElementById('denoiserJsonFile').files[0];
        const weightsFile = document.getElementById('denoiserWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both denoiser model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading denoiser model...');
            
            if (this.denoiserModel) {
                this.denoiserModel.dispose();
            }
            
            this.denoiserModel = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            this.showStatus('Denoiser model loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load denoiser model: ${error.message}`);
        }
    }

    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        if (this.denoiserModel) {
            this.denoiserModel.dispose();
            this.denoiserModel = null;
        }
        
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
        
        // Принудительная сборка мусора
        tf.engine().startScope();
        tf.engine().endScope();
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    renderPreview(images, predicted, trueLabels, indices) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        for (let i = 0; i < images.shape[0]; i++) {
            const item = document.createElement('div');
            item.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            canvas.width = 84;
            canvas.height = 84;
            
            const label = document.createElement('div');
            const isCorrect = predicted[i] === trueLabels[i];
            label.className = isCorrect ? 'correct' : 'wrong';
            label.textContent = `Pred: ${predicted[i]} | True: ${trueLabels[i]}`;
            
            const imageTensor = images.slice([i, 0, 0, 0], [1, 28, 28, 1]).squeeze().mul(255);
            this.drawTensorToCanvas(imageTensor, canvas);
            imageTensor.dispose();
            
            item.appendChild(canvas);
            item.appendChild(label);
            container.appendChild(item);
        }
    }

    clearPreview() {
        document.getElementById('previewContainer').innerHTML = '';
    }

    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        statusEl.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        
        if (!this.model && !this.denoiserModel) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
            return;
        }
        
        let classifierInfo = 'Not loaded';
        let denoiserInfo = 'Not loaded';
        
        if (this.model) {
            const layers = this.model.layers.length;
            classifierInfo = `Loaded (${layers} layers)`;
        }
        
        if (this.denoiserModel) {
            const layers = this.denoiserModel.layers.length;
            denoiserInfo = `Loaded (${layers} layers)`;
        }
        
        infoEl.innerHTML = `
            <h3>Model Info</h3>
            <p>Classifier: ${classifierInfo}</p>
            <p>Denoiser: ${denoiserInfo}</p>
        `;
    }

    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
    }

    showError(message) {
        this.showStatus(`ERROR: ${message}`);
        console.error(message);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
