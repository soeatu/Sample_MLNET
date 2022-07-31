using Sample_MLNET;

//.txtパス
string dataSetFilePath = @"C:.\DataSet\PetImages\_petImageFileList.txt";

//パス一覧の読み込む
var imagefilePaths = File.ReadAllLines(dataSetFilePath, System.Text.Encoding.Default);

//データセットの作成
var petDataSet = imagefilePaths.Select(path =>
    new PetData()
    {
        //ファイル名に品種名が含まれているので、品種名を切り出してBreedに設定
        Breed = path.Substring(path.LastIndexOf('\\') + 1,
        path.LastIndexOf('_') - path.LastIndexOf('\\') - 1),
        ImageFilePath = path
    });
