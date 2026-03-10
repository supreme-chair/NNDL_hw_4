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
        
        // Используем подмножество данных для скорости
        const numSamples = Math.min(2000, this.trainData.xs.shape[0]);
        
        // В data-loader.js данные уже нормализованы делением на 255
        // Поэтому используем их как есть
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
        
        // Добавляем валидацию данных
        const trainXValues = await trainXs.mean().data();
        const trainYValues = await trainYs.mean().data();
        this.showStatus(`Data stats - X mean: ${trainXValues[0].toFixed(4)}, Y mean: ${trainYValues[0].toFixed(4)}`);
        
        await this.model.fit(trainXs, trainYs, {
            epochs: 5,
            batchSize: 64,
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
    
    // Конволюционные слои с меньшим количеством фильтров
    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        inputShape: [28, 28, 1],
        kernelInitializer: 'glorotUniform'  // Меняем инициализацию
    }));
    
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }));
    
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    
    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu',
        kernelInitializer: 'glorotUniform'
    }));
    
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
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
        
        // Используем подмножество
        const numSamples = Math.min(1000, this.trainData.xs.shape[0]);
        
        // Данные уже нормализованы в data-loader.js
        const cleanImages = this.trainData.xs.slice([0, 0, 0, 0], [numSamples, 28, 28, 1]);
        
        // Создаем зашумленные версии
        const noise = tf.randomNormal([numSamples, 28, 28, 1], 0, 0.2);
        const noisyImages = tf.tidy(() => {
            return cleanImages.add(noise).clipByValue(0, 1);
        });
        
        // Преобразуем в плоские векторы для dense слоев
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
