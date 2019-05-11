#include <iostream>
#include <fdeep/fdeep.hpp>
using namespace std;

struct InputData {
    float leg_2_DPFTau_2016_v1tauVSall;
    float leg_2_byCombinedIsolationDeltaBetaCorrRaw3Hits;
    float leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2;
    float leg_2_chargedIsoPtSum;
    float leg_2_decayDistMag;
    float leg_2_decayMode;
    float leg_2_deepTau2017v1tauVSall;
    float leg_2_deepTau2017v1tauVSjet;
    float leg_2_dxy;
    float leg_2_dxy_Sig;
    float leg_2_eRatio;
    float leg_2_flightLengthSig;
    float leg_2_gjAngleDiff;
    float leg_2_hasSecondaryVertex;
    float leg_2_ip3d;
    float leg_2_nPhoton;
    float leg_2_neutralIsoPtSum;
    float leg_2_photonPtSumOutsideSignalCone;
    float leg_2_ptWeightedDetaStrip;
    float leg_2_ptWeightedDphiStrip;
    float leg_2_ptWeightedDrIsolation;
    float leg_2_ptWeightedDrSignal;
    float leg_2_puCorrPtSum;
};

class MLEvaluator
{
public:
    MLEvaluator() = default;

    virtual float evaluateOneObs(const InputData & OneObservation) = 0;

    vector<float> evaluate(const vector<InputData> & observations){
        vector<float> preds;
        for (InputData obs: observations){
            preds.push_back(evaluateOneObs(obs));
        }
        return preds;
    }
};


class MLEvaluatorOnlyDisc: public MLEvaluator
{
public:
    fdeep::model model;

    explicit MLEvaluatorOnlyDisc(string & directory): model(fdeep::load_model(directory)) {};

    float evaluateOneObs(const InputData & OneObservation) override
    {
        auto obs = {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 4), {
            OneObservation.leg_2_DPFTau_2016_v1tauVSall,
            OneObservation.leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2,
            OneObservation.leg_2_deepTau2017v1tauVSall,
            OneObservation.leg_2_deepTau2017v1tauVSjet,
        })};
        fdeep::tensor5 pred =  model.predict(obs)[0];
        return pred.get(0, 0, 0, 0, 0);

    }
};


class MLEvaluatorWholeData: public MLEvaluator
{
public:
    fdeep::model model;

    explicit MLEvaluatorWholeData(string & directory): model(fdeep::load_model(directory)) {};

    float evaluateOneObs(const InputData & OneObservation) override
    {
        auto obs = {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 23), {
                OneObservation.leg_2_DPFTau_2016_v1tauVSall,
                OneObservation.leg_2_byCombinedIsolationDeltaBetaCorrRaw3Hits,
                OneObservation.leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2,
                OneObservation.leg_2_chargedIsoPtSum,
                OneObservation.leg_2_decayDistMag,
                OneObservation.leg_2_decayMode,
                OneObservation.leg_2_deepTau2017v1tauVSall,
                OneObservation.leg_2_deepTau2017v1tauVSjet,
                OneObservation.leg_2_dxy,
                OneObservation.leg_2_dxy_Sig,
                OneObservation.leg_2_eRatio,
                OneObservation.leg_2_flightLengthSig,
                OneObservation.leg_2_gjAngleDiff,
                OneObservation.leg_2_hasSecondaryVertex,
                OneObservation.leg_2_ip3d,
                OneObservation.leg_2_nPhoton,
                OneObservation.leg_2_neutralIsoPtSum,
                OneObservation.leg_2_photonPtSumOutsideSignalCone,
                OneObservation.leg_2_ptWeightedDetaStrip,
                OneObservation.leg_2_ptWeightedDphiStrip,
                OneObservation.leg_2_ptWeightedDrIsolation,
                OneObservation.leg_2_ptWeightedDrSignal,
                OneObservation.leg_2_puCorrPtSum,
        })};
        fdeep::tensor5 pred =  model.predict(obs)[0];
        return pred.get(0, 0, 0, 0, 0);
    }
};

class MLEvaluatorWithoutDisc: public MLEvaluator
{
public:
    fdeep::model model;

    explicit MLEvaluatorWithoutDisc(string & directory): model(fdeep::load_model(directory)) {};

    float evaluateOneObs(const InputData & OneObservation) override
    {
        auto obs = {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 19), {
                OneObservation.leg_2_byCombinedIsolationDeltaBetaCorrRaw3Hits,
                OneObservation.leg_2_chargedIsoPtSum,
                OneObservation.leg_2_decayDistMag,
                OneObservation.leg_2_decayMode,
                OneObservation.leg_2_dxy,
                OneObservation.leg_2_dxy_Sig,
                OneObservation.leg_2_eRatio,
                OneObservation.leg_2_flightLengthSig,
                OneObservation.leg_2_gjAngleDiff,
                OneObservation.leg_2_hasSecondaryVertex,
                OneObservation.leg_2_ip3d,
                OneObservation.leg_2_nPhoton,
                OneObservation.leg_2_neutralIsoPtSum,
                OneObservation.leg_2_photonPtSumOutsideSignalCone,
                OneObservation.leg_2_ptWeightedDetaStrip,
                OneObservation.leg_2_ptWeightedDphiStrip,
                OneObservation.leg_2_ptWeightedDrIsolation,
                OneObservation.leg_2_ptWeightedDrSignal,
                OneObservation.leg_2_puCorrPtSum,
        })};
        fdeep::tensor5 pred =  model.predict(obs)[0];
        return pred.get(0, 0, 0, 0, 0);
    }
};

class MLEvaluatorWithoutbyCombined: public MLEvaluator
{
public:
    fdeep::model model;

    explicit MLEvaluatorWithoutbyCombined(string & directory): model(fdeep::load_model(directory)) {};

    float evaluateOneObs(const InputData & OneObservation) override
    {
        auto obs = {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 22), {
                OneObservation.leg_2_DPFTau_2016_v1tauVSall,
                OneObservation.leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2,
                OneObservation.leg_2_chargedIsoPtSum,
                OneObservation.leg_2_decayDistMag,
                OneObservation.leg_2_decayMode,
                OneObservation.leg_2_deepTau2017v1tauVSall,
                OneObservation.leg_2_deepTau2017v1tauVSjet,
                OneObservation.leg_2_dxy,
                OneObservation.leg_2_dxy_Sig,
                OneObservation.leg_2_eRatio,
                OneObservation.leg_2_flightLengthSig,
                OneObservation.leg_2_gjAngleDiff,
                OneObservation.leg_2_hasSecondaryVertex,
                OneObservation.leg_2_ip3d,
                OneObservation.leg_2_nPhoton,
                OneObservation.leg_2_neutralIsoPtSum,
                OneObservation.leg_2_photonPtSumOutsideSignalCone,
                OneObservation.leg_2_ptWeightedDetaStrip,
                OneObservation.leg_2_ptWeightedDphiStrip,
                OneObservation.leg_2_ptWeightedDrIsolation,
                OneObservation.leg_2_ptWeightedDrSignal,
                OneObservation.leg_2_puCorrPtSum,
        })};
        fdeep::tensor5 pred =  model.predict(obs)[0];
        return pred.get(0, 0, 0, 0, 0);
    }
};


class MLEvaluatorXGBWholeData: public MLEvaluator
{
public:
    vector<float, allocator<float>> (*predict_func)(vector<float> &);

    explicit MLEvaluatorXGBWholeData(vector<float, allocator<float>> (*func)(vector<float> &)):
    predict_func(func) {};

    float softmax(vector<float> prediction){
        return exp(prediction[1])/(exp(prediction[1]) + exp(prediction[0]));
    }

    float evaluateOneObs(const InputData & OneObservation) override
    {
        vector<float> obs = {
                OneObservation.leg_2_DPFTau_2016_v1tauVSall,
                OneObservation.leg_2_byCombinedIsolationDeltaBetaCorrRaw3Hits,
                OneObservation.leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2,
                OneObservation.leg_2_chargedIsoPtSum,
                OneObservation.leg_2_decayDistMag,
                OneObservation.leg_2_decayMode,
                OneObservation.leg_2_deepTau2017v1tauVSall,
                OneObservation.leg_2_deepTau2017v1tauVSjet,
                OneObservation.leg_2_dxy,
                OneObservation.leg_2_dxy_Sig,
                OneObservation.leg_2_eRatio,
                OneObservation.leg_2_flightLengthSig,
                OneObservation.leg_2_gjAngleDiff,
                OneObservation.leg_2_hasSecondaryVertex,
                OneObservation.leg_2_ip3d,
                OneObservation.leg_2_nPhoton,
                OneObservation.leg_2_neutralIsoPtSum,
                OneObservation.leg_2_photonPtSumOutsideSignalCone,
                OneObservation.leg_2_ptWeightedDetaStrip,
                OneObservation.leg_2_ptWeightedDphiStrip,
                OneObservation.leg_2_ptWeightedDrIsolation,
                OneObservation.leg_2_ptWeightedDrSignal,
                OneObservation.leg_2_puCorrPtSum,
        };

        return softmax(predict_func(obs));
    }
};


class MLEvaluatorXGBWithoutDisc: public MLEvaluator
{
public:
    vector<float, allocator<float>> (*predict_func)(vector<float> &);

    explicit MLEvaluatorXGBWithoutDisc(vector<float, allocator<float>> (*func)(vector<float> &)):
            predict_func(func) {};

    float softmax(vector<float> prediction){
        return exp(prediction[1])/(exp(prediction[1]) + exp(prediction[0]));
    }

    float evaluateOneObs(const InputData & OneObservation) override
    {
        vector<float> obs = {
                OneObservation.leg_2_byCombinedIsolationDeltaBetaCorrRaw3Hits,
                OneObservation.leg_2_chargedIsoPtSum,
                OneObservation.leg_2_decayDistMag,
                OneObservation.leg_2_decayMode,
                OneObservation.leg_2_dxy,
                OneObservation.leg_2_dxy_Sig,
                OneObservation.leg_2_eRatio,
                OneObservation.leg_2_flightLengthSig,
                OneObservation.leg_2_gjAngleDiff,
                OneObservation.leg_2_hasSecondaryVertex,
                OneObservation.leg_2_ip3d,
                OneObservation.leg_2_nPhoton,
                OneObservation.leg_2_neutralIsoPtSum,
                OneObservation.leg_2_photonPtSumOutsideSignalCone,
                OneObservation.leg_2_ptWeightedDetaStrip,
                OneObservation.leg_2_ptWeightedDphiStrip,
                OneObservation.leg_2_ptWeightedDrIsolation,
                OneObservation.leg_2_ptWeightedDrSignal,
                OneObservation.leg_2_puCorrPtSum,
        };

        return softmax(predict_func(obs));
    }
};


class MLEvaluatorXGBWithoutbyCombined: public MLEvaluator
{
public:
    vector<float, allocator<float>> (*predict_func)(vector<float> &);

    explicit MLEvaluatorXGBWithoutbyCombined(vector<float, allocator<float>> (*func)(vector<float> &)):
            predict_func(func) {};

    float softmax(vector<float> prediction){
        return exp(prediction[1])/(exp(prediction[1]) + exp(prediction[0]));
    }

    float evaluateOneObs(const InputData & OneObservation) override
    {
        vector<float> obs = {
                OneObservation.leg_2_DPFTau_2016_v1tauVSall,
                OneObservation.leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2,
                OneObservation.leg_2_chargedIsoPtSum,
                OneObservation.leg_2_decayDistMag,
                OneObservation.leg_2_decayMode,
                OneObservation.leg_2_deepTau2017v1tauVSall,
                OneObservation.leg_2_deepTau2017v1tauVSjet,
                OneObservation.leg_2_dxy,
                OneObservation.leg_2_dxy_Sig,
                OneObservation.leg_2_eRatio,
                OneObservation.leg_2_flightLengthSig,
                OneObservation.leg_2_gjAngleDiff,
                OneObservation.leg_2_hasSecondaryVertex,
                OneObservation.leg_2_ip3d,
                OneObservation.leg_2_nPhoton,
                OneObservation.leg_2_neutralIsoPtSum,
                OneObservation.leg_2_photonPtSumOutsideSignalCone,
                OneObservation.leg_2_ptWeightedDetaStrip,
                OneObservation.leg_2_ptWeightedDphiStrip,
                OneObservation.leg_2_ptWeightedDrIsolation,
                OneObservation.leg_2_ptWeightedDrSignal,
                OneObservation.leg_2_puCorrPtSum,
        };

        return softmax(predict_func(obs));
    }
};

#include "xgboost_whole_data.cpp"
#include "xgboost_without_disc.cpp"
#include "xgboost_without_byCombined.cpp"

int main()
{
    string file = "/home/aga/Fizyka/licencjat/test/new_data_without_byCombined.json";
    string &file_ref = file;
    MLEvaluatorWithoutbyCombined ml(file_ref);
    vector<InputData> observations = {
            {0.0025482773780822754,482.3755798339844,-0.9917095899581909,163.88214111328125,0.005424702074378729,10.0,0.016672516241669655,0.017106447368860245,0.0004330444207880646,0.489096075296402,0.19491860270500183,-0.057426899671554565,2.4541208744049072,1.0,-0.008946210145950317,19.0,325.7962951660156,0.0,0.0885210707783699,0.2695022523403168,0.29661449790000916,0.0,36.5142822265625},
            {0.5269338488578796,1.4638240337371826,0.9471954703330994,1.4638240337371826,0.0,0.0,0.961125373840332,0.9713723659515381,0.0006646728725172579,2.349895477294922,0.6053470969200134,-9.899999618530273,-999.0,0.0,0.0039803991094231606,2.0,0.0,1.724774718284607,0.035292744636535645,0.24770480394363403,0.0,0.05746985226869583,19.400352478027344},
            {0.0,6.82457971572876,-0.5493193864822388,6.82457971572876,0.0,1.0,0.5901840329170227,0.5907617807388306,0.00013847350783180445,0.1386643648147583,0.772677481174469,-9.899999618530273,-999.0,0.0,0.004970621783286333,4.0,0.0,2.0840399265289307,0.05460011959075928,0.08469649404287338,0.0,0.04667908325791359,34.71544647216797},
    };
    auto result = ml.evaluate(observations);
    std::cout << result[0] << ' ' << result[1] << ' ' << result[2] << std::endl;

    MLEvaluatorXGBWithoutbyCombined model(xgb_classify_without_byCombined);
    vector<float> pred = model.evaluate(observations);
    std::cout << pred[0] << ' ' << pred[1] << ' ' << pred[2] << endl;
}