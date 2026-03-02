param(
    [ValidateSet("cpu", "gpu")]
    [string]$Device = "cpu",
    [int]$GpuId = 0,
    [int]$Epoch = 30,
    [int]$MoeNum = 8
)

$dataset = "wadi"
$seed = 5
$batchSize = 32
$slideWin = 5
$dim = 64
$outLayerNum = 1
$slideStride = 1
$topk = 5
$outLayerInterDim = 128
$valRatio = 0.2
$decay = 0
$pathPattern = $dataset
$comment = "wadi-default"
$report = "best"

if ($Device -eq "cpu") {
    python main.py `
        -dataset $dataset `
        -save_path_pattern $pathPattern `
        -slide_stride $slideStride `
        -slide_win $slideWin `
        -batch $batchSize `
        -epoch $Epoch `
        -comment $comment `
        -random_seed $seed `
        -decay $decay `
        -dim $dim `
        -out_layer_num $outLayerNum `
        -out_layer_inter_dim $outLayerInterDim `
        -val_ratio $valRatio `
        -report $report `
        -topk $topk `
        -moe_num $MoeNum `
        -device cpu
} else {
    $env:CUDA_VISIBLE_DEVICES = "$GpuId"
    python main.py `
        -dataset $dataset `
        -save_path_pattern $pathPattern `
        -slide_stride $slideStride `
        -slide_win $slideWin `
        -batch $batchSize `
        -epoch $Epoch `
        -comment $comment `
        -random_seed $seed `
        -decay $decay `
        -dim $dim `
        -out_layer_num $outLayerNum `
        -out_layer_inter_dim $outLayerInterDim `
        -val_ratio $valRatio `
        -report $report `
        -topk $topk `
        -moe_num $MoeNum
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}
