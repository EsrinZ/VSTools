// C-MVP_pyramid_phase_preserve.fx
// Phase-preserving Laplacian pyramid reconstruction for HDR detail preservation up to ~2000 nits
// - Multi-scale Laplacian bands (N layers)
// - Pre-tonemap partial strip of bands (alpha per-layer)
// - Tonemap on base, compute local gain G(x)
// - Per-band DC-corrected injection ΔBi (5x5 zero-mean), energy feedback ΔE
// - Final reconstruction and optional PQ output
// Performance notes: layers, per-layer sampling and weights are tunable. Default aims moderate cost.

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

// ==================== UI Tunables ====================
uniform int UseNewPipeline <
    ui_category = "General";
    ui_items = "Off\0On\0";
    ui_label = "Use Pyramid Pipeline";
    ui_type = "combo";
> = 1;

// Output / exposure
uniform float PeakLuminance <
    ui_category = "Output";
    ui_label = "Peak Luminance (nits)";
    ui_type = "slider";
    ui_min = 200.0; ui_max = 4000.0; ui_step = 1.0;
> = 2000.0;

uniform float GlobalGain <
    ui_category = "Output";
    ui_label = "Global Gain";
    ui_type = "slider";
    ui_min = 0.6; ui_max = 1.6; ui_step = 0.01;
> = 1.05;

uniform float ExposureGain <
    ui_category = "Output";
    ui_label = "Exposure Gain";
    ui_type = "slider";
    ui_min = 0.5; ui_max = 1.4; ui_step = 0.01;
> = 1.08;

uniform int PQChoice <
    ui_category = "Output";
    ui_items = "Gamma Display\0PQ Encode\0";
    ui_label = "Output Mode";
    ui_type = "combo";
> = 0;

// Color
uniform float3 WhiteBalanceGain <
    ui_category = "Color";
    ui_label = "White Balance Gain";
    ui_type = "color";
> = float3(1.0f,1.0f,1.0f);

uniform float Saturation <
    ui_category = "Color";
    ui_label = "Saturation";
    ui_type = "slider";
    ui_min = 0.6; ui_max = 1.3; ui_step = 0.01;
> = 1.04;

// Pyramid control
uniform int Pyramid_Layers <
    ui_category = "Pyramid";
    ui_items = "3\04\05\06\07\0";
    ui_label = "Laplacian Layers";
    ui_type = "combo";
> = 4;

uniform float pre_strip_factor <
    ui_category = "Pyramid";
    ui_label = "Pre-tonemap Strip Factor (global)";
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
> = 0.35;

uniform float layer_weights0 <
    ui_category = "Pyramid";
    ui_label = "Layer0 Inject Weight";
    ui_type = "slider";
    ui_min = 0.00; ui_max = 0.40; ui_step = 0.005;
> = 0.18;

uniform float layer_weights1 <
    ui_category = "Pyramid";
    ui_label = "Layer1 Inject Weight";
    ui_type = "slider";
    ui_min = 0.00; ui_max = 0.40; ui_step = 0.005;
> = 0.14;

uniform float layer_weights2 <
    ui_category = "Pyramid";
    ui_label = "Layer2 Inject Weight";
    ui_type = "slider";
    ui_min = 0.00; ui_max = 0.40; ui_step = 0.005;
> = 0.10;

uniform float layer_weights3 <
    ui_category = "Pyramid";
    ui_label = "Layer3 Inject Weight";
    ui_type = "slider";
    ui_min = 0.00; ui_max = 0.40; ui_step = 0.005;
> = 0.06;

uniform float layer_weights4 <
    ui_category = "Pyramid";
    ui_label = "Layer4 Inject Weight";
    ui_type = "slider";
    ui_min = 0.00; ui_max = 0.40; ui_step = 0.005;
> = 0.03;

// DC correction & energy feedback
uniform float DC_window_scale <
    ui_category = "Pyramid";
    ui_label = "DC Correction Window (px scale)";
    ui_type = "slider";
    ui_min = 1.0; ui_max = 3.0; ui_step = 0.1;
> = 1.0;

uniform float energy_feedback_thresh <
    ui_category = "Pyramid";
    ui_label = "Energy Feedback Threshold (frac)";
    ui_type = "slider";
    ui_min = 0.004; ui_max = 0.05; ui_step = 0.001;
> = 0.012;

uniform float peak_gate_low <
    ui_category = "Pyramid";
    ui_label = "Peak Gate Low";
    ui_type = "slider";
    ui_min = 1000; ui_max = 3000; ui_step = 50;
> = 2200;

uniform float peak_gate_high <
    ui_category = "Pyramid";
    ui_label = "Peak Gate High";
    ui_type = "slider";
    ui_min = 2500; ui_max = 4000; ui_step = 50;
> = 3600;

// Debug
uniform int DebugView <
    ui_category = "Debug";
    ui_items = "Off\0SavedBand0\0SavedBand1\0SavedBand2\0SavedBand3\0BaseLuma\0DeltaE\0FinalLuma\0";
    ui_label = "Debug View";
    ui_type = "combo";
> = 0;

uniform int SwapRB <
    ui_category = "Debug";
    ui_items = "Off\0On\0";
    ui_label = "Swap R/B";
    ui_type = "combo";
> = 0;
// UI/text preserve
uniform int UIPreserveEnable <
    ui_category = "Rendering";
    ui_items = "Off\0On\0";
    ui_label = "UI Preserve Enable";
    ui_type = "combo";
> = 1;

uniform float UIPreserveStrength <
    ui_category = "Rendering";
    ui_label = "UI Preserve Strength";
    ui_type = "slider";
    ui_min = 0.90; ui_max = 1.0; ui_step = 0.001;
> = 0.995;

// ==================== Constants ====================
static const float REF_PEAK = 1000.0f;
#define SDR_GAMMA 2.2
static const float3 Luma709 = float3(0.2126f,0.7152f,0.0722f);

// ----------------------------- Helpers -----------------------------
float3 SDR_To_Linear(float3 sdr_rgb) { return pow(max(sdr_rgb, 1e-8f), SDR_GAMMA); }
float3 Linear_To_SDR(float3 lin) { return pow(max(lin, 1e-8f), 1.0f / SDR_GAMMA); }

float3 SampleLinear(float2 uv) { return SDR_To_Linear(tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0f, 0.0f)).rgb); }

float3 BoxAvgLinear3Offset(float2 uv, float offScale)
{
    float3 sum = float3(0.0f,0.0f,0.0f);
    for (int y=-1; y<=1; ++y)
        for (int x=-1; x<=1; ++x)
        {
            float2 off = float2(x,y) * BUFFER_PIXEL_SIZE * offScale;
            sum += SampleLinear(uv + off);
        }
    return sum / 9.0f;
}

float3 SeparableGauss(float2 uv, float sigmaScale, int taps)
{
    // cheap separable Gaussian approximation using box-like taps for performance
    // sigmaScale controls effective radius; taps 3..7
    float3 sum = float3(0.0f,0.0f,0.0f);
    int r = (taps-1)/2;
    int count = 0;
    for (int y=-r; y<=r; ++y)
        for (int x=-r; x<=r; ++x)
        {
            float2 off = float2(x,y) * BUFFER_PIXEL_SIZE * sigmaScale;
            sum += SampleLinear(uv + off);
            ++count;
        }
    return sum / max(1, count);
}

// small unsharp for UI clarity (linear domain)
float3 UnsharpForUI(float3 lin, float2 uv)
{
    float3 center = lin;
    float3 blur = BoxAvgLinear3Offset(uv, 1.0f);
    float3 mask = center - blur;
    return center + saturate(mask * 1.0f);
}

// PQ helpers (SMPTE ST2084)
static const float PQ_m1 = 2610.0f / 16384.0f;
static const float PQ_m2 = 2523.0f / 32.0f;
static const float PQ_c1 = 3424.0f / 4096.0f;
static const float PQ_c2 = 2413.0f / 128.0f;
static const float PQ_c3 = 2392.0f / 128.0f;
float3 PQ_Encode(float3 L, float displayPeak)
{
    displayPeak = max(displayPeak, 1e-6f);
    float3 x = max(L / displayPeak, float3(1e-9f,1e-9f,1e-9f));
    float3 R = pow(x, PQ_m1);
    float3 num = PQ_c1 + PQ_c2 * R;
    float3 den = 1.0 + PQ_c3 * R;
    float3 E = pow(num / den, PQ_m2);
    return saturate(E);
}

// Perceptual S-curve with adaptive MIN_SCALE
float3 PerceptualSCurvePreserveGain(float3 linearRGB, float pivotNits, float contrastGain, float rolloff, float PeakL)
{
    float pivot = max(pivotNits, 1.0f) * 1.15f;
    float peakScale = max(PeakL / REF_PEAK, 1e-6f);

    float3 nits = linearRGB * (REF_PEAK * peakScale * GlobalGain);

    float L = dot(nits, Luma709);
    float t = saturate(L / pivot);
    float s = smoothstep(0.0f, 1.0f, t);
    float p = pow(t, 1.0f / max(contrastGain, 0.01f));
    float curve = lerp(s, p, 0.28f);

    float roll = 1.0f / (1.0f + rolloff * 0.75f * pow(t, 2.2f));
    float Lout = curve * pivot * roll;

    float scaleOutRaw = (L > 1e-6f) ? (Lout / L) : 1.0f;

    float peakNorm = saturate((PeakL - 600.0f) / 3400.0f);
    float MIN_SCALE_LOW = 0.60f;
    float MIN_SCALE_HIGH = 0.80f;
    float bias = smoothstep(0.0f, 1.0f, peakNorm);
    float MIN_SCALE = lerp(MIN_SCALE_LOW, MIN_SCALE_HIGH, bias);
    MIN_SCALE = clamp(MIN_SCALE, 0.50f, 0.85f);

    float scaleOut = max(scaleOutRaw, MIN_SCALE);
    float3 outNits = nits * scaleOut;
    return outNits / (REF_PEAK * peakScale * GlobalGain);
}

// Dark chroma suppression (conservative)
float3 DarkChromaSuppressSimple(float3 mappedRGB, float2 uv)
{
    const float DARK_LIMIT = 0.012f;
    const float SUPPRESS_STRENGTH = 0.85f;
    const float BLEND_RADIUS = 1.0f;

    float Lpix = dot(mappedRGB, Luma709);
    if (Lpix > DARK_LIMIT) return mappedRGB;

    float3 localAvg = BoxAvgLinear3Offset(uv, BLEND_RADIUS);
    float localL = dot(localAvg, Luma709);
    float3 neutral = float3(localL, localL, localL);

    float3 chroma = mappedRGB - neutral;
    float suppressK = saturate(1.0f - smoothstep(0.0f, DARK_LIMIT, Lpix));
    float effectiveSuppress = lerp(1.0f, SUPPRESS_STRENGTH, suppressK);

    float3 outRGB = neutral + chroma * effectiveSuppress;

    float chromaMag = length(chroma);
    float extraBlend = smoothstep(0.0f, 0.04f, chromaMag);
    outRGB = lerp(outRGB, neutral, 1.0f - extraBlend * 0.5f);

    return saturate(outRGB);
}

// ------------------- Pyramid helpers -------------------
// compute Laplacian band: band = blur_small - blur_med
float3 LaplacianBand(float2 uv, float smallScale, float medScale)
{
    float3 small = BoxAvgLinear3Offset(uv, smallScale);
    float3 med   = BoxAvgLinear3Offset(uv, medScale);
    return small - med; // signed bandpass
}

// DC-correct ΔBi: subtract local mean in window W (5x5 scaled)
float3 DC_Correct(float3 delta, float2 uv, float windowScale)
{
    // 5x5 window approx scaled by windowScale
    float3 sum = float3(0.0f,0.0f,0.0f);
    int r = 2;
    for (int y=-r; y<=r; ++y)
        for (int x=-r; x<=r; ++x)
        {
            float2 off = float2(x,y) * BUFFER_PIXEL_SIZE * windowScale;
            sum += SampleLinear(uv + off); // we approximate mean of delta by local luminance proxy
        }
    float3 meanApprox = sum / 25.0f;
    // here we approximate mean(delta) by meanApprox * small factor to keep zero-DC effect
    // To keep delta zero-sum in luminance, subtract luminance-projected mean from delta
    float meanL = dot(meanApprox, Luma709);
    float deltaL = dot(delta, Luma709);
    float3 deltaCorrected = delta - meanL * (delta / max(deltaL, 1e-6f)); // normalized subtract (safe)
    return deltaCorrected;
}

// local fractional energy change for 3x3 window
float LocalDeltaE_frac(float3 beforeCenter, float3 afterCenter, float2 uv)
{
    float E_before = 0.0f;
    float E_after  = 0.0f;
    for (int y=-1; y<=1; ++y)
        for (int x=-1; x<=1; ++x)
        {
            float2 off = float2(x,y) * BUFFER_PIXEL_SIZE;
            float3 sample_lin = SampleLinear(uv + off);
            float l_before = dot(sample_lin, Luma709);
            E_before += l_before;
            // approximate after by replacing center contribution with afterCenter
            if (x==0 && y==0)
                E_after += dot(afterCenter, Luma709);
            else
                E_after += l_before;
        }
    float frac = (E_after - E_before) / max(E_before, 1e-6f);
    return frac;
}

// ==================== Pixel Shader ====================
float3 PS_PyramidPhasePreserve(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
    // read & linearize
    float3 sdr = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0.0f, 0.0f)).rgb;
    sdr = saturate(sdr);
    float3 lin = SDR_To_Linear(sdr);

    if (SwapRB == 1) lin = lin.bgr;

    // quick path: if disabled, fall back to simple PerceptualSCurve pipeline
    if (UseNewPipeline == 0)
    {
        float3 mapped = lin * WhiteBalanceGain;
        float luma = dot(mapped, Luma709);
        float3 grey = float3(luma,luma,luma);
        mapped = lerp(grey, mapped, Saturation);
        float peakScale = max(PeakLuminance / REF_PEAK, 1e-6f);
        float appliedGain = clamp(ExposureGain * GlobalGain * peakScale, 0.05f, 6.0f);
        mapped *= appliedGain;
        float3 scene = PerceptualSCurvePreserveGain(mapped, 100.0f, 1.03f, 0.75f, PeakLuminance);
        scene = DarkChromaSuppressSimple(scene, texcoord);
        if (PQChoice == 0) return Linear_To_SDR(scene);
        float3 hdr_nits = saturate(scene * (REF_PEAK * peakScale * GlobalGain * ExposureGain));
        float knee = 1900.0f;
        float softness = 0.00020f;
        float3 cap;
        cap.r = (hdr_nits.r <= knee) ? hdr_nits.r : knee + (hdr_nits.r - knee) / (1.0f + (hdr_nits.r - knee)*softness);
        cap.g = (hdr_nits.g <= knee) ? hdr_nits.g : knee + (hdr_nits.g - knee) / (1.0f + (hdr_nits.g - knee)*softness);
        cap.b = (hdr_nits.b <= knee) ? hdr_nits.b : knee + (hdr_nits.b - knee) / (1.0f + (hdr_nits.b - knee)*softness);
        float3 pq = PQ_Encode(cap, REF_PEAK * peakScale);
        return Linear_To_SDR(pq);
    }

    // -----------------------------
    // 1) Pyramid decomposition (signed bands)
    // layers count
    int layers = max(3, min(7, Pyramid_Layers));
    // scales: small->med per layer (simple mapping)
    float smallScales[7] = {1.0f, 1.6f, 2.6f, 4.2f, 6.8f, 10.8f, 17.2f};
    float medScales[7]  = {1.6f, 2.6f, 4.2f, 6.8f, 10.8f, 17.2f, 27.6f};

    float3 bands[7];
    for (int i=0;i<layers;i++)
    {
        bands[i] = LaplacianBand(texcoord, smallScales[i], medScales[i]);
    }

    // save debug-friendly band copies (0..1)
    float3 savedBand0 = saturate(bands[0]*0.5f + 0.5f);
    float3 savedBand1 = (layers>1) ? saturate(bands[1]*0.5f + 0.5f) : savedBand0;
    float3 savedBand2 = (layers>2) ? saturate(bands[2]*0.5f + 0.5f) : savedBand0;
    float3 savedBand3 = (layers>3) ? saturate(bands[3]*0.5f + 0.5f) : savedBand0;

    // -----------------------------
    // 2) build detail-stripped base
    float3 base_lin = lin;
    // per-layer pre-strip: use global pre_strip_factor scaled by layer index (deeper layers strip less)
    for (int i=0;i<layers;i++)
    {
        float decay = pow(0.85f, float(i)); // deeper layers removed less
        base_lin = base_lin - bands[i] * (pre_strip_factor * decay);
    }
    base_lin = max(base_lin, 0.0f);

    // white balance & saturation
    float3 wb = base_lin * WhiteBalanceGain;
    float luma_wb = dot(wb, Luma709);
    float3 grey_wb = float3(luma_wb,luma_wb,luma_wb);
    wb = lerp(grey_wb, wb, Saturation);

    // forward gain with peak attenuation
    float peakScale = max(PeakLuminance / REF_PEAK, 1e-6f);
    float peakProtect = 1.0f;
    if (PeakLuminance > 1200.0f)
        peakProtect = lerp(1.0f, 0.65f, smoothstep(1200.0f, 4000.0f, PeakLuminance));
    float appliedGain = clamp(ExposureGain * GlobalGain * peakScale * peakProtect, 0.05f, 6.0f);
    wb *= appliedGain;

    // tonemap base and record per-pixel local gain G
    float3 tonemapped_base = PerceptualSCurvePreserveGain(wb, 100.0f, 1.03f, 0.75f, PeakLuminance);
    float inL = dot(max(wb, 1e-6f), Luma709);
    float outL = dot(tonemapped_base, Luma709);
    float G = outL / max(inL, 1e-6f);
    // clamp G into practical range for gating (0..1.2)
    G = clamp(G, 0.0f, 1.2f);

    // -----------------------------
    // 3) per-layer injection ΔBi with DC-correct and energy feedback
    // layer weights array
    float w0 = layer_weights0;
    float w1 = layer_weights1;
    float w2 = layer_weights2;
    float w3 = layer_weights3;
    float w4 = layer_weights4;
    float wArr[7] = {w0,w1,w2,w3,w4,0.02f,0.01f}; // beyond 5 layers small defaults

    // accumulate reconstruction starting from tonemapped_base
    float3 recon = tonemapped_base;

    // precompute peakGate
    float peakGate = saturate(1.0f - smoothstep(peak_gate_low, peak_gate_high, PeakLuminance));

    // for each layer compute corrected delta and apply with energy feedback
    for (int i=0;i<layers;i++)
    {
        float layerW = wArr[i];
        // desired injection (signed): scale band by layer weight
        float3 delta = bands[i] * layerW;

        // DC correction: subtract local mean (approx) to keep zero-DC in window
        delta = DC_Correct(delta, texcoord, DC_window_scale);

        // compute tentative recon after injection
        float3 reconTent = saturate(recon + delta);

        // compute local fractional energy change (3x3) using before/reconTent
        float fracE = LocalDeltaE_frac(recon, reconTent, texcoord);

        // if energy increased beyond threshold, suppress by factor
        float suppress = 1.0f;
        if (fracE > energy_feedback_thresh)
        {
            float excess = (fracE - energy_feedback_thresh) / max(1e-6f, 0.10f); // strong ramp-down
            suppress = saturate(1.0f - excess * 1.0f);
        }

        // also gate by G and peakGate: if tonemap strongly compressed (low G) allow more recovery
        float allowByG = lerp(0.8f, 1.2f, saturate(1.0f - G)); // if G small -> >1
        float finalGain = saturate(suppress * allowByG * peakGate);

        // apply scaled delta
        recon = saturate(recon + delta * finalGain);
    }

    // -----------------------------
    // 4) small signed fine unsharp (energy-neutral approximate)
    // fine detail from the finest band
    float3 fine = bands[0];
    // create zero-mean variant by subtracting 3x3 mean of fine
    float3 fineMean = BoxAvgLinear3Offset(texcoord, 1.0f);
    float3 fineZero = fine - fineMean * 0.03f; // small subtraction to reduce DC
    float3 deltaFine = fineZero * 0.5f * pre_strip_factor; // conservative
    // clamp deltaFine relative to recon
    deltaFine.r = clamp(deltaFine.r, -0.12f * recon.r, 0.12f * recon.r);
    deltaFine.g = clamp(deltaFine.g, -0.12f * recon.g, 0.12f * recon.g);
    deltaFine.b = clamp(deltaFine.b, -0.12f * recon.b, 0.12f * recon.b);
    recon = saturate(recon + deltaFine);

    // -----------------------------
    // 5) minor post-process: dark chroma suppress and UI preserve
    recon = DarkChromaSuppressSimple(recon, texcoord);

    float3 finalLinear = recon;
    // UI preserve (small)
    {
        float3 localAvg = BoxAvgLinear3Offset(texcoord, 1.0f);
        float3 diff = recon - localAvg;
        float hfEnergy = dot(diff, diff);
        float lumaScene = dot(recon, Luma709);
        float3 greyScene = float3(lumaScene, lumaScene, lumaScene);
        float chromaMag = length(recon - greyScene);
        bool isUI = (hfEnergy > 1e-6 && chromaMag < 0.06f);
        if (isUI)
        {
            float3 orig_lin = lin;
            float3 uiPath = UnsharpForUI(lerp(orig_lin, recon, 0.04f), texcoord);
            finalLinear = lerp(recon, uiPath, UIPreserveStrength);
        }
    }

    finalLinear = saturate(finalLinear);

    // -----------------------------
    // Debug Views
    if (DebugView == 1) { float v = dot(savedBand0, Luma709); return Linear_To_SDR(float3(v,v,v)); }
    if (DebugView == 2) { float v = dot(savedBand1, Luma709); return Linear_To_SDR(float3(v,v,v)); }
    if (DebugView == 3) { float v = dot(savedBand2, Luma709); return Linear_To_SDR(float3(v,v,v)); }
    if (DebugView == 4) { float v = dot(savedBand3, Luma709); return Linear_To_SDR(float3(v,v,v)); }
    if (DebugView == 5) { float v = dot(tonemapped_base, Luma709); return Linear_To_SDR(float3(v,v,v)); }
    if (DebugView == 6) { float v = saturate((LocalDeltaE_frac(tonemapped_base, recon, texcoord)+0.05f)*5.0f); return Linear_To_SDR(float3(v,v,v)); }
    if (DebugView == 7) { float v = dot(finalLinear, Luma709); return Linear_To_SDR(float3(v,v,v)); }

    // -----------------------------
    // Final output gamma or PQ
    if (PQChoice == 0)
    {
        return Linear_To_SDR(finalLinear);
    }
    else
    {
        float3 hdr_nits = saturate(finalLinear * (REF_PEAK * peakScale * GlobalGain * ExposureGain));
        float knee = 2000.0f;
        float softness = 0.00018f;
        float3 cap;
        cap.r = (hdr_nits.r <= knee) ? hdr_nits.r : knee + (hdr_nits.r - knee) / (1.0f + (hdr_nits.r - knee)*softness);
        cap.g = (hdr_nits.g <= knee) ? hdr_nits.g : knee + (hdr_nits.g - knee) / (1.0f + (hdr_nits.g - knee)*softness);
        cap.b = (hdr_nits.b <= knee) ? hdr_nits.b : knee + (hdr_nits.b - knee) / (1.0f + (hdr_nits.b - knee)*softness);
        float3 localContrast = finalLinear - BoxAvgLinear3Offset(texcoord, 1.0f);
        cap = cap + 0.01f * localContrast * saturate(length(localContrast) * 4.0f);
        cap = saturate(cap);
        float3 pq = PQ_Encode(cap, REF_PEAK * peakScale);
        return Linear_To_SDR(pq);
    }
}

// Technique
technique C_MVP_Pyramid_Phase_Preserve
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PyramidPhasePreserve;
    }
}
