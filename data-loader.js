class MNISTDataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    // Разделяем на строки и удаляем пустые
                    const lines = content.split('\n').filter(line => line.trim() !== '');
                    
                    // Пропускаем заголовок если есть
                    const firstLine = lines[0].split(',');
                    const startIdx = (firstLine.length === 1 && firstLine[0] === 'label') ? 1 : 0;
                    
                    const labels = [];
                    const pixels = [];
                    
                    for (let i = startIdx; i < lines.length; i++) {
                        const line = lines[i].trim();
                        if (line === '') continue;
                        
                        const values = line.split(',').map(val => {
                            const num = Number(val);
                            return isNaN(num) ? 0 : num;
                        });
                        
                        // Проверяем правильность формата
                        if (values.length >= 785) {
                            labels.push(values[0]);
                            pixels.push(values.slice(1, 785));
                        } else if (values.length === 784) {
                            // Если нет метки, используем 0
                            labels.push(0);
                            pixels.push(values);
                        }
                    }
                    
                    if (labels.length === 0) {
                        reject(new Error('No valid data found in file'));
                        return;
                    }
                    
                    console.log(`Loaded ${labels.length} samples`);
                    console.log(`First label: ${labels[0]}`);
                    console.log(`First pixel value: ${pixels[0][0]}`);
                    
                    const xs = tf.tidy(() => {
                        // Нормализуем пиксели к [0, 1]
                        const tensor = tf.tensor2d(pixels);
                        return tensor.div(255).reshape([labels.length, 28, 28, 1]);
                    });
                    
                    const ys = tf.tidy(() => {
                        return tf.oneHot(labels, 10);
                    });
                    
                    resolve({ xs, ys, count: labels.length });
                    
                } catch (error) {
                    console.error('Parse error:', error);
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numVal = Math.floor(xs.shape[0] * valRatio);
            const numTrain = xs.shape[0] - numVal;
            
            const trainXs = xs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]);
            const trainYs = ys.slice([0, 0], [numTrain, 10]);
            
            const valXs = xs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]);
            const valYs = ys.slice([numTrain, 0], [numVal, 10]);
            
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const total = xs.shape[0];
            const indices = [];
            for (let i = 0; i < k; i++) {
                indices.push(Math.floor(Math.random() * total));
            }
            
            const batchXs = tf.gather(xs, indices);
            const batchYs = tf.gather(ys, indices);
            
            return { batchXs, batchYs, indices };
        });
    }

    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        return tf.tidy(() => {
            const ctx = canvas.getContext('2d');
            
            // Получаем данные и убеждаемся что они в правильном диапазоне
            const data = tensor.reshape([28, 28]).mul(255).dataSync();
            
            // Создаем ImageData
            const imgData = new ImageData(28, 28);
            
            for (let i = 0; i < 784; i++) {
                const val = Math.min(255, Math.max(0, Math.round(data[i])));
                imgData.data[i * 4] = val;
                imgData.data[i * 4 + 1] = val;
                imgData.data[i * 4 + 2] = val;
                imgData.data[i * 4 + 3] = 255;
            }
            
            // Масштабируем для отображения
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            ctx.imageSmoothingEnabled = false;
            
            // Создаем временный canvas для масштабирования
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imgData, 0, 0);
            
            ctx.drawImage(tempCanvas, 0, 0, 28 * scale, 28 * scale);
        });
    }

    dispose() {
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
    }
}