import Vue from "vue";
import Router from "vue-router";
import Draw from "./views/Draw.vue";
import store from "./store.js";
import axios from "axios";

Vue.use(Router);

let router = new Router({
  mode: "history",
  base: process.env.BASE_URL,
  routes: [
    {
      path: "/",
      name: "Draw",
      component: Draw,
      meta: {
        requiresAuth: false
      }
    },
  ],
  scrollBehavior() {
    return { x: 0, y: 0 };
  }
});


export default router;
