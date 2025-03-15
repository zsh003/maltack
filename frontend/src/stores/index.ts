// // store/index.ts
// import { createStore } from 'vuex';
// import { AnalysisResult } from '../views/UploadView.vue';
//
// export default createStore<{
//     analysisResult: AnalysisResult | null
// }>({
//     state() {
//         return {
//             analysisResult: null,
//         };
//     },
//     mutations: {
//         setAnalysisResult(state, payload) {
//             state.analysisResult = payload;
//         },
//     },
//     actions: {
//         updateAnalysisResult({ commit }, result) {
//             commit('setAnalysisResult', result);
//         },
//     },
// });
