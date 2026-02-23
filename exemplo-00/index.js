import tf from '@tensorflow/tfjs-node';

async function trainModel(xs, ys) {
    // Criamos um modelo sequencial simples
    const model = tf.sequential();
    // Primeira camada da rede:
    // entrada 7 posições (idade normalizada, cores, localizações)
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' })); // Camada oculta com 10 neurônios
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // Camada de saída com 3 neurônios (para 3 categorias)

    // Compilamos o modelo com otimizador e função de perda

    // loss: categoricalCrossentropy é usada para problemas de classificação com múltiplas classes (one-hot encoded)
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamos o modelo
    await model.fit(xs, ys, { 
        verbose: 0,
        epochs: 100,
        shuffle: true,
        callbacks:{
            onEpochEnd: (epoch, log) => console.log(
                `Epoch: ${epoch}: loss = ${log.loss}` 
            )
        } 
    }
    );

    return model;
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

model.predict(inputXs).print()