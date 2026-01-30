import Vue from "vue";
import vuetify from "@/plugins/vuetify";
import VueClipboard from "vue-clipboard2";
import App from "./App.vue";
import router from "./router";
import store from "./store";

Vue.config.productionTip = false;

new Vue({
  vuetify,
  router,
  store,
  render: h => h(App)
}).$mount("#app");

Vue.use(VueClipboard);
