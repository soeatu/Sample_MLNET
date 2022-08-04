using Sample_MLNET;
using System.Collections.Generic;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Vision;
using Microsoft.ML.Data;

#region データセットの読み込み
//.txtパス
string dataSetFilePath = @"C:.\DataSet\PetImages\_petImageFileList.txt";

//パス一覧の読み込む
EncodingProvider provider = System.Text.CodePagesEncodingProvider.Instance;
Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
var sjis = Encoding.GetEncoding("shift_jis");
var imagefilePaths = File.ReadAllLines(dataSetFilePath, sjis);

// データセットの作成
var petDataSet = imagefilePaths.Select(path =>
    new PetData()
    {
        // ファイル名に品種名が含まれるので、品種名を切り出して Breed に設定
        Breed = path.Substring(path.LastIndexOf('\\') + 1,
        path.LastIndexOf('_') - path.LastIndexOf('\\') - 1),
        ImageFilePath = path
    });
#endregion

#region　データセットの加工
//コンテキストの生成
MLContext mLContext = new MLContext(seed:1);
//データのロード
IDataView petDataView = mLContext.Data.LoadFromEnumerable(petDataSet);
//データセットをシャッフル
IDataView shuffledpetDataView = mLContext.Data.ShuffleRows(petDataView);

//データの前加工
IDataView transformedDataView = mLContext.Transforms.Conversion.MapValueToKey(
    //品質文字列を数値に変換して別名をlabelとする
    inputColumnName:nameof(PetData.Breed),
    outputColumnName: "Label",
    keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
    .Append(mLContext.Transforms.LoadRawImageBytes(
        // パスから画像をロード
        inputColumnName: nameof(PetData.ImageFilePath),
        imageFolder: null,
        outputColumnName: "RawImageBytes"))
    .Fit(shuffledpetDataView)   //　データセット用に Transformer を生成 
    .Transform(shuffledpetDataView); // Transformer をデータセットに適用

// データセットを学習データ、検証データ、評価データに分割
// データセットを 7:3 に分割
var trainValidationTestSplit = mLContext.Data.TrainTestSplit(transformedDataView, testFraction: 0.3);
// 検証/評価データセットを 8:2 に分割
var validationTestSplit = mLContext.Data.TrainTestSplit(trainValidationTestSplit.TestSet, testFraction: 0.2);

// データセットの 70% を学習データとする
IDataView trainDataView = trainValidationTestSplit.TrainSet;
// データセットの 24% を検証データとする
IDataView validationDataView = validationTestSplit.TrainSet;
// データセットの 6% を評価データとする
IDataView testDataView = validationTestSplit.TestSet;
#endregion

#region 学習の定義
// 学習の定義
var trainer = mLContext.MulticlassClassification.Trainers.ImageClassification(
    new ImageClassificationTrainer.Options()
    {
        LabelColumnName = "Label", //ラベル列
        FeatureColumnName = "RawImageBytes", // 特徴列
        Arch = ImageClassificationTrainer.Architecture.ResnetV250, //転移学習モデルの選択
        Epoch = 200,
        BatchSize = 10,
        LearningRate = 0.01f,
        ValidationSet = validationDataView, // 検証データを設定
        MetricsCallback = (metrics) => Console.WriteLine(metrics),
        WorkspacePath = @".\Workspace",
    })
    .Append(mLContext.Transforms.Conversion.MapKeyToValue(
        // 推論結果のラベルを数値から品種文字列に変換
        inputColumnName: "PredictedLabel",
        outputColumnName: "PredictedBreed"));

// 学習の実行
ITransformer trainedModel = trainer.Fit(trainDataView);

// 学習モデルをファイルに保存
string modelFilePath = $@".\model{DateTimeOffset.Now:yyyyMMddHmmss}.zip";
mLContext.Model.Save(trainedModel, trainDataView.Schema, modelFilePath);
#endregion

#region　学習モデルの評価
// テストデータで推論を実行
IDataView testDataPredictionsDataView = trainedModel.Transform(testDataView);
IEnumerable<PetDataPrediction> predictions = 
    mLContext.Data.CreateEnumerable<PetDataPrediction>(testDataPredictionsDataView, reuseRowObject: true).Take(10);
// テストデータでの推論結果をもとに評価指標を計算
var metrics = mLContext.MulticlassClassification.Evaluate(testDataPredictionsDataView);

// ラベルと品種文字列のキーバリューを取得
VBuffer<ReadOnlyMemory<char>> keyValues = default;
trainDataView.Schema["Label"].GetKeyValues(ref keyValues);


string testFilePath = $@".\test{DateTimeOffset.Now:yyyyMMddHHmmss}.html";

// HTML で評価結果を書き出し
using (var writer = new StreamWriter(testFilePath))
{
    writer.WriteLine($"<html><head><title>{Path.GetFileName(modelFilePath)}</title>");
    writer.WriteLine("<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css\" integrity=\"sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2\" crossorigin=\"anonymous\">");
    writer.WriteLine("</head><body>");

    writer.WriteLine($"<h1>Metrics for {Path.GetFileName(modelFilePath)}</h1>");
    // メトリックの書き出し
    writer.WriteLine("<div><table class=\"table table-striped\">");
    writer.WriteLine($"<tr><td>MicroAccuracy</td><td>{metrics.MicroAccuracy:0.000}</td></tr></tr>");
    writer.WriteLine($"<tr><td>MacroAccuracy</td><td>{metrics.MacroAccuracy:0.000}</td></tr></tr>");
    writer.WriteLine($"<tr><td>Precision</td><td>{metrics.ConfusionMatrix.PerClassPrecision.Average():0.000}</td></tr></tr>");
    writer.WriteLine($"<tr><td>Recall</td><td>{metrics.ConfusionMatrix.PerClassRecall.Average():0.000}</td></tr></tr>");
    writer.WriteLine($"<tr><td>LogLoss</td><td>{metrics.LogLoss:0.000}</td></tr></tr>");
    writer.WriteLine($"<tr><td>LogLossReduction</td><td>{metrics.LogLossReduction:0.000}</td></tr></tr>");

    // クラス毎の適合率
    writer.WriteLine("<tr><td>PerClassPrecision</td><td>");
    metrics.ConfusionMatrix.PerClassPrecision
    .Select((p, i) => (Precision: p, Index: i))
    .ToList()
    .ForEach(p =>
        writer.WriteLine($"{keyValues.GetItemOrDefault(p.Index)}: {p.Precision:0.000}<br />"));
    writer.WriteLine("</td></tr>");

    // クラス毎の再現率
    writer.WriteLine("<tr><td>PerClassRecall</td><td>");
    metrics.ConfusionMatrix.PerClassRecall
    .Select((p, i) => (Recall: p, Index: i))
    .ToList()
    .ForEach(p =>
        writer.WriteLine($"{keyValues.GetItemOrDefault(p.Index)}: {p.Recall:0.000}<br />"));
    writer.WriteLine("</td></tr></table></div>");

    // 評価データ毎の分類結果
    writer.WriteLine($"<h1>Predictions</h1>");
    writer.WriteLine($"<div><table class=\"table table-bordered\">");

    foreach (var prediction in predictions)
    {
        writer.WriteLine($"<tr><td>");
        // 画像ファイル名
        writer.WriteLine($"{Path.GetFileName(prediction.ImageFilePath)}<br />");
        // 正解ラベル
        writer.WriteLine($"Actual Value: {prediction.Breed}<br />");
        // 推論結果
        writer.WriteLine($"Predicted Value: {prediction.PredictedBreed}<br />");
        // 画像
        writer.WriteLine($"<img class=\"img-fluid\" src=\"{prediction.ImageFilePath}\" /></td>");
        // クラス毎の推論結果
        writer.WriteLine($"<td>");
        prediction.Score.Select((s, i) => (Index: i, Label: keyValues.GetItemOrDefault(i), Score: s))
        .OrderByDescending(c => c.Score)
        .Take(10) // 上位 10 件
        .ToList()
        .ForEach(c =>
        {
            writer.WriteLine($"{c.Label}: {c.Score:P}<br />");
        });

        writer.WriteLine("</td></tr>");
    }

    writer.WriteLine("</table></div>");
    writer.WriteLine("</body></html>");
}

#endregion