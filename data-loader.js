// В data-loader.js, метод loadCSVFile должен быть таким:

async loadCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = (event) => {
            try {
                const content = event.target.result;
                const lines = content.split('\n').filter(line => line.trim() !== '');
                
                const labels = [];
                const pixels = [];
                
                for (const line of lines) {
                    const values = line.split(',').map(Number);
                    if (values.length !== 785) continue; // label + 784 pixels
                    
                    labels.push(values[0]);
                    pixels.push(values.slice(1));
                }
                
                if (labels.length === 0) {
                    reject(new Error('No valid data found in file'));
                    return;
                }
                
                // Нормализуем пиксели к [0, 1] (это правильно)
                const xs = tf.tidy(() => {
                    return tf.tensor2d(pixels)
                        .div(255)  // Это нормализация
                        .reshape([labels.length, 28, 28, 1]);
                });
                
                // One-hot encode labels
                const ys = tf.tidy(() => {
                    return tf.oneHot(labels, 10);
                });
                
                resolve({ xs, ys, count: labels.length });
            } catch (error) {
                reject(error);
            }
        };
        
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}
