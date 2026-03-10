class MNISTApp {

constructor(){
    this.dataLoader = new MNISTDataLoader()

    this.modelMax = null
    this.modelAvg = null

    this.trainData = null
    this.testData = null

    this.initializeUI()
}

initializeUI(){

    document.getElementById('loadDataBtn').addEventListener('click',()=>this.onLoadData())
    document.getElementById('trainBtn').addEventListener('click',()=>this.onTrain())
    document.getElementById('evaluateBtn').addEventListener('click',()=>this.onEvaluate())
    document.getElementById('testFiveBtn').addEventListener('click',()=>this.onTestFive())
    document.getElementById('saveModelBtn').addEventListener('click',()=>this.onSaveDownload())
    document.getElementById('resetBtn').addEventListener('click',()=>this.onReset())

}

showStatus(msg){

    const logs=document.getElementById("trainingLogs")

    const entry=document.createElement("div")

    entry.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`

    logs.appendChild(entry)

}

showError(msg){

    this.showStatus("ERROR: "+msg)

}

async onLoadData(){

    const trainFile=document.getElementById("trainFile").files[0]
    const testFile=document.getElementById("testFile").files[0]

    if(!trainFile || !testFile){
        this.showError("Select train and test CSV")
        return
    }

    this.showStatus("Loading train data...")
    this.trainData=await this.dataLoader.loadTrainFromFiles(trainFile)

    this.showStatus("Loading test data...")
    this.testData=await this.dataLoader.loadTestFromFiles(testFile)

    this.showStatus("Data loaded")

}

createAutoencoder(poolType){

    const input=tf.input({shape:[28,28,1]})

    let x=tf.layers.conv2d({
        filters:16,
        kernelSize:3,
        activation:'relu',
        padding:'same'
    }).apply(input)

    if(poolType==="max")
        x=tf.layers.maxPooling2d({poolSize:[2,2]}).apply(x)
    else
        x=tf.layers.averagePooling2d({poolSize:[2,2]}).apply(x)

    x=tf.layers.conv2d({
        filters:8,
        kernelSize:3,
        activation:'relu',
        padding:'same'
    }).apply(x)

    x=tf.layers.upSampling2d({size:[2,2]}).apply(x)

    const output=tf.layers.conv2d({
        filters:1,
        kernelSize:3,
        activation:'sigmoid',
        padding:'same'
    }).apply(x)

    const model=tf.model({inputs:input,outputs:output})

    model.compile({
        optimizer:'adam',
        loss:'binaryCrossentropy'
    })

    return model
}

async onTrain(){

    if(!this.trainData){
        this.showError("Load data first")
        return
    }

    this.showStatus("Preparing training")

    const {trainXs,valXs}=this.dataLoader.splitTrainVal(
        this.trainData.xs,
        this.trainData.ys,
        0.1
    )

    const trainSubset=trainXs.slice([0,0,0,0],[6000,28,28,1])
    const valSubset=valXs.slice([0,0,0,0],[1000,28,28,1])

    const noisyTrain=addNoise(trainSubset)
    const noisyVal=addNoise(valSubset)

    this.modelMax=this.createAutoencoder("max")
    this.modelAvg=this.createAutoencoder("avg")

    this.showStatus("Training MAX model")

    await this.modelMax.fit(noisyTrain,trainSubset,{
        epochs:5,
        batchSize:64,
        validationData:[noisyVal,valSubset]
    })

    this.showStatus("Training AVG model")

    await this.modelAvg.fit(noisyTrain,trainSubset,{
        epochs:5,
        batchSize:64,
        validationData:[noisyVal,valSubset]
    })

    trainXs.dispose()
    valXs.dispose()
    trainSubset.dispose()
    valSubset.dispose()
    noisyTrain.dispose()
    noisyVal.dispose()

    this.showStatus("Training complete")

}

async onEvaluate(){

    if(!this.modelMax || !this.modelAvg){
        this.showError("Train model first")
        return
    }

    const noisy=addNoise(this.testData.xs)

    const lossMax=this.modelMax.evaluate(noisy,this.testData.xs)
    const lossAvg=this.modelAvg.evaluate(noisy,this.testData.xs)

    const valMax=(await lossMax.data())[0]
    const valAvg=(await lossAvg.data())[0]

    this.showStatus(`MAX loss: ${valMax.toFixed(5)}`)
    this.showStatus(`AVG loss: ${valAvg.toFixed(5)}`)

    noisy.dispose()

}

async onTestFive(){

    if(!this.modelMax || !this.modelAvg){
        this.showError("Train model first")
        return
    }

    const container=document.getElementById("previewContainer")
    container.innerHTML=""

    const {batchXs}=this.dataLoader.getRandomTestBatch(
        this.testData.xs,
        this.testData.ys,
        5
    )

    const noisy=addNoise(batchXs)

    const maxPred=this.modelMax.predict(noisy)
    const avgPred=this.modelAvg.predict(noisy)

    for(let i=0;i<5;i++){

        const row=document.createElement("div")
        row.style.display="flex"
        row.style.gap="20px"

        const c1=document.createElement("canvas")
        const c2=document.createElement("canvas")
        const c3=document.createElement("canvas")
        const c4=document.createElement("canvas")

        const orig=batchXs.slice([i,0,0,0],[1,28,28,1]).squeeze()
        const noisyImg=noisy.slice([i,0,0,0],[1,28,28,1]).squeeze()
        const maxImg=maxPred.slice([i,0,0,0],[1,28,28,1]).squeeze()
        const avgImg=avgPred.slice([i,0,0,0],[1,28,28,1]).squeeze()

        this.dataLoader.draw28x28ToCanvas(orig,c1,4)
        this.dataLoader.draw28x28ToCanvas(noisyImg,c2,4)
        this.dataLoader.draw28x28ToCanvas(maxImg,c3,4)
        this.dataLoader.draw28x28ToCanvas(avgImg,c4,4)

        row.appendChild(c1)
        row.appendChild(c2)
        row.appendChild(c3)
        row.appendChild(c4)

        container.appendChild(row)

        orig.dispose()
        noisyImg.dispose()
        maxImg.dispose()
        avgImg.dispose()
    }

    batchXs.dispose()
    noisy.dispose()
    maxPred.dispose()
    avgPred.dispose()

}

async onSaveDownload(){

    if(!this.modelMax){
        this.showError("Train model first")
        return
    }

    await this.modelMax.save('downloads://mnist-denoiser-max')
    await this.modelAvg.save('downloads://mnist-denoiser-avg')

    this.showStatus("Models saved")

}

onReset(){

    if(this.modelMax) this.modelMax.dispose()
    if(this.modelAvg) this.modelAvg.dispose()

    this.dataLoader.dispose()

    this.modelMax=null
    this.modelAvg=null
    this.trainData=null
    this.testData=null

    document.getElementById("previewContainer").innerHTML=""

    this.showStatus("Reset done")

}

}

function addNoise(images,noiseFactor=0.3){

return tf.tidy(()=>{

    const noise=tf.randomNormal(images.shape)

    const noisy=images.add(noise.mul(noiseFactor))

    return noisy.clipByValue(0,1)

})

}

document.addEventListener("DOMContentLoaded",()=>{

    new MNISTApp()

})
