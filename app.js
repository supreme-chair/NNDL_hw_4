class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        this.initializeUI();
    }

    initializeUI() {
        // Bind button events
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
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
            
            // Split training data
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );

            // Create or get model
            if (!this.model) {
                this.model = this.createModel();
                this.updateModelInfo();
            }

            // Train with tfjs-vis callbacks
            const startTime = Date.now();
            const history = await this.model.fit(trainXs, trainYs, {
                epochs: 5,
                batchSize: 128,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss', 'val_loss', 'acc', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            const duration = (Date.now() - startTime) / 1000;
            const bestValAcc = Math.max(...history.history.val_acc);
            
            this.showStatus(`Training completed in ${duration.toFixed(1)}s. Best val_acc: ${bestValAcc.toFixed(4)}`);
            
            // Clean up
            trainXs.dispose();
            trainYs.dispose();
            valXs.dispose();
            valYs.dispose();
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
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
            
            const testXs = this.testData.xs;
            const testYs = this.testData.ys;
            
            // Get predictions
            const predictions = this.model.predict(testXs);
            const predictedLabels = predictions.argMax(-1);
            const trueLabels = testYs.argMax(-1);
            
            // Calculate accuracy
            const accuracy = await this.calculateAccuracy(predictedLabels, trueLabels);
            
            // Create confusion matrix data
            const confusionMatrix = await this.createConfusionMatrix(predictedLabels, trueLabels);
            
            // Show metrics in visor
            const metricsContainer = { name: 'Test Metrics', tab: 'Evaluation' };
            
            // Overall accuracy
            tfvis.show.modelSummary(metricsContainer, this.model);
            tfvis.show.perClassAccuracy(metricsContainer, 
                { values: this.calculatePerClassAccuracy(confusionMatrix) }, 
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            );
            
            // Confusion matrix
            tfvis.render.confusionMatrix(metricsContainer, {
                values: confusionMatrix,
                tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            });
            
            this.showStatus(`Test accuracy: ${(accuracy * 100).toFixed(2)}%`);
            
            // Clean up
            predictions.dispose();
            predictedLabels.dispose();
            trueLabels.dispose();
            
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
            
            // Clean up
            predictions.dispose();
            predictedLabels.dispose();
            trueLabels.dispose();
            batchXs.dispose();
            batchYs.dispose();
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    async onSaveDownload() {
        if (!this.model) {
            this.showError('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-cnn');
            this.showStatus('Model saved successfully!');
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
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
            this.showStatus('Loading model...');
            
            // Dispose old model if exists
            if (this.model) {
                this.model.dispose();
            }
            
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            this.showStatus('Model loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    createModel() {
        const model = tf.sequential();
        
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 }));
        model.add(tf.layers.flatten());
        
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async calculateAccuracy(predicted, trueLabels) {
        const equals = predicted.equal(trueLabels);
        const accuracy = equals.mean();
        const result = await accuracy.data();
        equals.dispose();
        accuracy.dispose();
        return result[0];
    }

    async createConfusionMatrix(predicted, trueLabels) {
        const predArray = await predicted.array();
        const trueArray = await trueLabels.array();
        
        const matrix = Array(10).fill().map(() => Array(10).fill(0));
        
        for (let i = 0; i < predArray.length; i++) {
            const pred = predArray[i];
            const trueVal = trueArray[i];
            matrix[trueVal][pred]++;
        }
        
        return matrix;
    }

    calculatePerClassAccuracy(confusionMatrix) {
        return confusionMatrix.map((row, i) => {
            const correct = row[i];
            const total = row.reduce((sum, val) => sum + val, 0);
            return total > 0 ? correct / total : 0;
        });
    }

    renderPreview(images, predicted, trueLabels, indices) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        // Convert tensor to array for processing
        const imageArray = images.arraySync();
        
        imageArray.forEach((image, i) => {
            const item = document.createElement('div');
            item.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            const label = document.createElement('div');
            
            const isCorrect = predicted[i] === trueLabels[i];
            label.className = isCorrect ? 'correct' : 'wrong';
            label.textContent = `Pred: ${predicted[i]} | True: ${trueLabels[i]}`;
            
            // Draw image to canvas
            this.dataLoader.draw28x28ToCanvas(tf.tensor(image), canvas, 4);
            
            item.appendChild(canvas);
            item.appendChild(label);
            container.appendChild(item);
        });
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
        
        if (!this.model) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
            return;
        }
        
        let totalParams = 0;
        this.model.layers.forEach(layer => {
            layer.getWeights().forEach(weight => {
                totalParams += weight.size;
            });
        });
        
        infoEl.innerHTML = `
            <h3>Model Info</h3>
            <p>Layers: ${this.model.layers.length}</p>
            <p>Total parameters: ${totalParams.toLocaleString()}</p>
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

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
