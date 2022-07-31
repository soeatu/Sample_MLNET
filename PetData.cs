using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Sample_MLNET
{
    /// <summary>
    /// 学習データカラムの定義
    /// </summary>
    public class PetData
    {
        public string? Breed { get; set; }
        public string? ImageFilePath {  get; set; }
    }

    public class PetDataPrediction : PetData
    {
        public string? PredictedBreed { get; set; }
        public float[]? Score { get; set; }
    }
}
