// MiniHDRSim.fx
// 极简可编译 ReShade FX（从零开始，便于逐步扩展）
// 仅保留最小流程： read linear -> base/detail -> controlled gain -> tonemap + small perceptual boost -> detail inject -> soft highlight -> output

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "DrawText.fxh"


// 调试开关与屏幕尺寸（调试完删除）
#define SHOW_DEBUG  1                 // 1 打开调试显示

//uniform int SHOW_DEBUG = 0; // 1 打开调试显示
// -------------------- 参数（ReShade 兼容 uniform 声明） --------------------
// 全局物理增益（整体曝光基线，过大易溢出）建议 1.0（范围 0.25..32）
uniform float GlobalGain <
    ui_label = "全局物理增益(GlobalGain)";
    ui_type = "slider";
    ui_min = 0.25; ui_max = 32.0; ui_step = 0.1;
> = 1.0;        

// 曝光偏移（EV 单位，每 +1 倍亮度）建议 0（范围 -4..+4） 
uniform float ExposureEV <
    ui_label = "曝光偏移(ExposureEV)";
    ui_type = "slider";
    ui_min = -4.0; ui_max = 4.0; ui_step = 0.1;
> = 0.1;         

// 感知微提升（在 tone map 前的小幅对数域提升，用于增强可见度）建议 0.3..0.8（范围 -2..+2）
uniform float PerceptualEV <
    ui_label = "感知微提升(PerceptualEV)";
    ui_type = "slider";
    ui_min = -2.0; ui_max = 2.0; ui_step = 0.1;
> = 0.3;       

// 细节注入强度（高频纹理/锐度控制）建议 0.3..1.0（范围 0..2）
uniform float DetailGain <
    ui_label = "细节注入强度(DetailGain)";
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.1;
> = 1.0;         

// 软高光提升（让高光“有能量”而非直接翻满）建议 0.10..0.30（范围 0..1）
uniform float HighlightBoost <
    ui_label = "软高光提升(HighlightBoost)";
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.91;

// 暗场抬升，单位 EV，建议范围 -1.0 .. 2.0    
uniform float ShadowBoostEV <
    ui_label = "暗场抬升(ShadowBoostEV)";
    ui_type = "slider";
    ui_min = -1.0; ui_max = 2.0; ui_step = 0.1;
> = -0.4;   

// 暗场影响截止 (nits-like), 建议 0.005..0.05
uniform float ShadowKnee <
    ui_label = "暗场影响截止(ShadowKnee)";
    ui_type = "slider";
    ui_min = 0.001; ui_max = 0.1; ui_step = 0.001;
> = 0.1;  

// 暗场最大线性增益上限，防止噪点放大
uniform float ShadowMaxGain <
    ui_label = "暗场最大线性增益上限(ShadowMaxGain)";
    ui_type = "slider";
    ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
> = 1.0;   

// >1 增强对比 1.0 = 无变化
uniform float ContrastGain <
    ui_label = "增强对比(ContrastGain)";
    ui_type = "slider";
    ui_min = 0.5; ui_max = 2.0; ui_step = 0.01;
> = 1.02;    

// 对比中心（线性域），0.18 常用
uniform float ContrastPivot <
    ui_label = "对比中心(ContrastPivot)";
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.5;   

// 抬暗部基线，0..0.1 之间微调
uniform float BlackLift <
    ui_label = "抬暗部基线(BlackLift)";
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
> = 0.00;       

// 1.0 = 无，>1 允许极亮处有更大头部但随之 soft-clamp
uniform float WhiteSoftClip <
    ui_label = "极亮处面积(WhiteSoftClip)";
    ui_type = "slider";
    ui_min = 1.0; ui_max = 1.2; ui_step = 0.01;
> = 1.00; 
// 饱和度与颜色调整（极简中文注释）
uniform float Saturation <
    ui_label = "饱和度(Saturation)";
    ui_type  = "slider";
    ui_min   = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 0.8;    // 1.0 = 无变化, >1 增强, 0 = 灰度

uniform float Vibrance <
    ui_label = "活力 (Vibrance)";
    ui_type  = "slider";
    ui_min   = -1.0; ui_max = 2.0; ui_step = 0.01;
> = 2.00;    // 正值增强中低饱和，负值降低鲜艳度

uniform float3 ColorGain <
    ui_label = "颜色增益 (ColorGain)";
    ui_type  = "color";
    ui_min   = 0.0; ui_max = 2.0; ui_step = 0.01;
> = float3(1.0, 1.0, 1.0); // 用于白平衡或色偏校正，例如暖色增红: (1.05,1.0,0.98)
  
uniform float HighlightThreshold <
    ui_label = "高光阈值(HighlightThreshold)";
    ui_type  = "slider";
    ui_min   = 0.0; ui_max = 2.0; ui_step = 0.001;
> = 0.47;    // 高光起始阈值（线性域, >1 常表示高于显示白），默认 11.6

uniform float HighlightBoostLocal <
    ui_label = "高光局部膨胀(HighlightBoostLocal)";
    ui_type  = "slider";
    ui_min   = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.35;    // 高光局部膨胀强度（0..1）

uniform float HighlightSoftKnee <
    ui_label = "高光平滑过渡(HighlightSoftKnee)";
    ui_type  = "slider";
    ui_min   = 0.0; ui_max = 1.0; ui_step = 0.001;
> = 0.0;     // 高光软阈宽度（0..1）

uniform float BloomStrength <
    ui_label = "Bloom 强度(BloomStrength)";
    ui_type  = "slider";
    ui_min   = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.18;    // Bloom 加法强度

uniform float BloomRadius <
    ui_label = "Bloom 半径(BloomRadius)";
    ui_type  = "slider";
    ui_min   = 0.5; ui_max = 8.0; ui_step = 0.1;
> = 2.0;     // Bloom 采样半径规模（像素级因子）

uniform float HighlightPreserve <
    ui_label = "高光保色率(HighlightPreserve)";
    ui_type  = "slider";
    ui_min   = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.0;     // 高光颜色保留率（0..1，1 完全保色）



// -------------------- 常量 --------------------
static const float3 Luma709 = float3(0.2126, 0.7152, 0.0722);
static const float EPS = 1e-6f;

// -------------------- 辅助函数（极简） --------------------
// SRGB -> linear (assume input is sRGB)
float3 SDR_To_Linear(float3 c)
{
    return pow(max(c, EPS), 2.2); // simple inverse gamma to linear
}

// linear -> sRGB
float3 Linear_To_SDR(float3 c)
{
    return pow(max(c, 0.0f), 1.0 / 2.2);
}

// sample ReShade backbuffer (linearized)
float3 SampleLinear(float2 uv)
{
    float3 s = tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0f, 0.0f)).rgb;
    return SDR_To_Linear(s);
}

// 3x3 box lowpass (per-channel), using SampleLinear
float3 BoxAvg3(in float2 uv, in float scale)
{
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float2 off = BUFFER_PIXEL_SIZE * scale;
    for (int y=-1; y<=1; y++)
        for (int x=-1; x<=1; x++)
            sum += SampleLinear(uv + float2(x,y) * off);
    return sum / 9.0f;
}

// 简单 Reinhard tone map in linear domain
float3 Tone_Reinhard(float3 x)
{
    return x / (1.0f + x);
}

// soft highlight lift （极简，受控）
float3 HighlightSoft(float3 v, float knee, float boost)
{
    float3 n = v / max(knee, 1e-6f);
    float3 add = pow(1.0f + n, 1.0f + boost) - (1.0f + n);
    return v + add * knee * 0.5f;
}

// detail scale 根据 base luminance 轻微衰减（高亮略减）
float DetailScale(float baseL)
{
    float s = 1.0f / (1.0f + 0.6f * saturate(baseL * 0.1f));
    return clamp(s, 0.6f, 1.4f);
}
// Shadow lift: 在 log 域对低亮度做受控抬升
float3 ShadowLift(float3 nits, float knee, float boostEV, float maxGain)
{
    // nits: 线性域值（base_nits 或 tone_input）
    // knee: 控制影响截止（越小作用越集中在非常暗处）
    // boostEV: 抬升量（EV）；正值增加暗部亮度，负值减小
    // maxGain: 线性增益上限，防止极端放大噪声

    // per-channel safe transform: operate on luminance then modulate color
    float L = dot(nits, Luma709);
    // compute target gain in linear domain from EV
    float targetGain = pow(2.0, boostEV);
    // weight: 1 at L==0, smoothly fall to 0 above knee*10
    float w = saturate( smoothstep(0.0, knee * 10.0, knee * 10.0) ); // placeholder stable fallback
    // better: weight decreases with luminance (stronger at darker)
    w = saturate( 1.0 - smoothstep(knee * 0.5, knee * 20.0, L) );
    // compute final gain, clamp to maxGain
    float gain = lerp(1.0, targetGain, w);
    gain = min(gain, maxGain);
    // apply color-preserving gain (scale rgb)
    return nits * gain;
}
// simple contrast around pivot with soft clipping for highlights and small lift for blacks
float3 ContrastEnhance(float3 col, float gain, float pivot, float blackLift, float whiteClip)
{
    // apply black lift first (keeps shadows readable)
    col = col + blackLift;

    // linear contrast around pivot
    col = (col - pivot) * gain + pivot;

    // soft clamp on highlights to avoid harsh saturation
    // soft knee using smoothstep-like polynomial
    float3 over = max(col - 1.0, 0.0);
    float3 knee = smoothstep(0.0, 1.0, over / max(whiteClip - 1.0, 1e-6));
    col = lerp(col, 1.0 + (over / (1.0 + over)), knee); // soften roll-off

    // ensure non-negative
    return max(col, 0.0);
}
// 计算像素色度并做饱和度变换（保色相）
float3 ApplySaturation(float3 rgb, float sat)
{
    // 灰度基底（保留亮度）
    float lum = dot(rgb, Luma709);
    float3 gray = float3(lum, lum, lum);
    return lerp(gray, rgb, sat); // sat=1 保持原色；sat=0 灰度
}

// Vibrance: 针对低饱和像素更强，避免高饱和溢出
float3 ApplyVibrance(float3 rgb, float vib)
{
    // 计算像素饱和度距离（简单度量）
    float maxc = max(max(rgb.r, rgb.g), rgb.b);
    float minc = min(min(rgb.r, rgb.g), rgb.b);
    float saturation = (maxc - minc) / max(maxc, 1e-6);
    // 以非线性权重对中低饱和区域增强
    float factor = saturate( vib * (1.0 - saturation) );
    // 以细微的饱和度提升实现“活力”
    float3 outx = ApplySaturation(rgb, 1.0 + factor * 0.5);
    return outx;
}

// 三通道增益（直接线性乘）
float3 ApplyColorGain(float3 rgb, float3 gain)
{
    return rgb * gain;
}
// 高光掩码（平滑软阈值）
float HighlightMask(float luma, float threshold, float softKnee)
{
    // soft knee: 0..1 -> 转换为宽度
    float knee = threshold * softKnee;
    // smooth step from (threshold - knee) .. (threshold + knee)
    return saturate( smoothstep(threshold - knee, threshold + knee, luma) );
}

// 局部高光膨胀（在 linear/映射域对高亮做非线性膨胀，保色）
float3 LocalHighlightBoost_Sensitive(float3 color, float mask, float threshold, float localBoost, float preserve)
{
    if (mask <= 1e-5) return color;
    float L = dot(color, Luma709);
    // ratio 表示亮度相对阈值（避免除零）
    float ratio = L / max(threshold, 1e-6);
    // 用 log2 或幂放大差异，参数 localBoost 控制强度
    float amp = pow(max(ratio, 0.0), 0.5 + localBoost); // 0.5..1.5 的灵活响应
    amp = lerp(1.0, amp, mask); // 只在掩码处生效
    // color-preserving add: 把亮度按 amp 增加一部分
    float3 boosted = color * amp * (1.0 - preserve) + color * preserve;
    // 限幅，避免过度
    return min(boosted, color + 2.0 * mask * localBoost);
}

// 极简 bloom：稀疏采样 box blur 的低成本近似（采样偏移随 BloomRadius）
float3 CheapBloom(float2 uv, float2 texel, float radius, float maskScale)
{
    float3 sum = float3(0.0,0.0,0.0);
    // 5 taps cross-shaped for low cost
    sum += tex2Dlod(ReShade::BackBuffer, float4(uv + float2(0,0)*texel*radius,0,0)).rgb * 0.4;
    sum += tex2Dlod(ReShade::BackBuffer, float4(uv + float2(radius,0)*texel,0,0)).rgb * 0.15;
    sum += tex2Dlod(ReShade::BackBuffer, float4(uv + float2(-radius,0)*texel,0,0)).rgb * 0.15;
    sum += tex2Dlod(ReShade::BackBuffer, float4(uv + float2(0,radius)*texel,0,0)).rgb * 0.15;
    sum += tex2Dlod(ReShade::BackBuffer, float4(uv + float2(0,-radius)*texel,0,0)).rgb * 0.15;
    return sum * maskScale;
}
// 使用 tone 映射后或 log(luminance) 做 brightpass，返回 0..1 掩码并增强响应
float BrightMask(float luminance, float threshold, float softKnee)
{
    // 掩码从 threshold 开始上升，到 threshold + softKnee 达到 1
    return smoothstep(threshold, threshold + softKnee, luminance);
}




float4 PS_Main(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    float3 sdr = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0.0, 0.0)).rgb;
    sdr = saturate(sdr);
    float3 lin = SDR_To_Linear(sdr);

    float3 base = BoxAvg3(texcoord, 1.0);
    float3 detail = lin - base;

    float exposureMultiplier = pow(2.0, ExposureEV);
    float appliedGain = clamp(GlobalGain * exposureMultiplier, 1e-4, 32.0);

    float perceptualBoost = pow(2.0, clamp(PerceptualEV, -2.0, 2.0));
    float3 base_nits = base * appliedGain;
    float3 detail_nits = detail * appliedGain;

    base_nits = ShadowLift(base_nits, ShadowKnee, ShadowBoostEV, ShadowMaxGain);

    float3 tone_input = base_nits * perceptualBoost;
    float3 tone_base  = Tone_Reinhard(tone_input);

    float baseL = dot(base_nits, Luma709);

    float dscale = DetailScale(baseL) * DetailGain;
    dscale *= saturate( lerp(0.9, 1.0, smoothstep(0.02, 0.5, baseL)) );
    float3 detail_mapped = Tone_Reinhard(detail_nits * dscale);
    float3 recon = tone_base + detail_mapped;

    // --- 高光掩码（使用 tone-mapped 亮度）
    float toneL = dot(recon, Luma709);
    float bmask = BrightMask(toneL, HighlightThreshold, HighlightSoftKnee);

    // --- 局部高光膨胀
    float3 localBoosted = LocalHighlightBoost_Sensitive(recon, bmask, HighlightThreshold, HighlightBoostLocal, HighlightPreserve);
    recon = lerp(recon, localBoosted, bmask);

    // --- bloom（cheap cross blur）
    float2 texel = BUFFER_PIXEL_SIZE;
    float r = max(1.0, BloomRadius);
    float3 bloom = float3(0.0, 0.0, 0.0);
    bloom += tex2Dlod(ReShade::BackBuffer, float4(texcoord,0,0)).rgb * 0.4;
    bloom += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(r,0)*texel,0,0)).rgb * 0.15;
    bloom += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(-r,0)*texel,0,0)).rgb * 0.15;
    bloom += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0,r)*texel,0,0)).rgb * 0.15;
    bloom += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(0,-r)*texel,0,0)).rgb * 0.15;
    bloom *= BloomStrength * bmask;
    recon += bloom;

    float knee = max(0.02, baseL * 0.5 + 0.02);
    recon = HighlightSoft(recon, knee, saturate(HighlightBoost));

    recon = ContrastEnhance(recon, ContrastGain, ContrastPivot, BlackLift, WhiteSoftClip);
    recon = ApplyColorGain(recon, ColorGain);
    if (abs(Vibrance) > 1e-5) recon = ApplyVibrance(recon, Vibrance);
    if (abs(Saturation - 1.0) > 1e-5) recon = ApplySaturation(recon, Saturation);

    recon = saturate(recon);
    float3 out_sdr = Linear_To_SDR(recon);
    return float4(out_sdr, 1.0);
}


// -------------------- Technique --------------------
technique Minimal_SDR2HDR
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_Main;
    }
}
