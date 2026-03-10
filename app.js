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
        
        const numSamples = Math.min(5000, this.trainData.xs.shape[0]);
        
        const xs = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
        const ys = this.trainData.ys.slice([0, 0], [numSamples, 10]);
        
        const min = await xs.min().data();
        const max = await xs.max().data();
        this.showStatus(`Data range: [${min[0].toFixed(3)}, ${max[0].toFixed(3)}]`);
        
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
        
        await this.model.fit(trainXs, trainYs, {
            epochs: 5,
            batchSize: 128,
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
        
        const numSamples = Math.min(1000, this.trainData.xs.shape[0]);
        
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
            epochs: 10,
            batchSize: 64,
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
