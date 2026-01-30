import Vue from "vue";
import Vuex from "vuex";
import axios from "axios";

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    status: "",
    token: localStorage.getItem("token") || "",
    username: localStorage.getItem("username") || "",
    level: Number(localStorage.getItem("level")) || 1,
    points: Number(localStorage.getItem("points")) || 0
  },
  mutations: {
    auth_request(state) {
      state.status = "loading";
    },
    auth_success(state, { token, username, level, points }) {
      state.status = "success";
      state.token = token;
      state.username = username;
      state.level = level;
      state.points = points;
    },
    auth_error(state) {
      state.status = "error";
    },
    logout(state) {
      state.status = "";
      state.token = "";
      state.username = "";
      state.level = 1;
      state.points = 0;
    },
    update_level_points(state, { level_new, points_new }) {
      state.level = level_new;
      state.points = points_new;
      localStorage.setItem("level", level_new);
      localStorage.setItem("points", points_new);
    }
  },
  actions: {
    login({ commit }, username_password) {
      return new Promise((resolve, reject) => {
        commit("auth_request");
        const config = { headers: { "Content-Type": "multipart/form-data" } };
        const fd = new FormData();
        fd.append("username", username_password.username);
        fd.append("password", username_password.password);
        axios
          .post(process.env.VUE_APP_API_URL+"/login", fd, config)
          .then(res => {
            if (res.data.success) {
              const token = res.data.token;
              const username = res.data.username;
              const level = res.data.level;
              const points = res.data.points;
              localStorage.setItem("token", token);
              localStorage.setItem("username", username);
              localStorage.setItem("level", level);
              localStorage.setItem("points", points);
              axios.defaults.headers.common["Authorization"] = token;
              commit("auth_success", { token, username, level, points });
            } else {
              commit("auth_error");
              localStorage.removeItem("token");
              localStorage.removeItem("username");
              localStorage.removeItem("level");
              localStorage.removeItem("points");
            }
            resolve(res);
          })
          .catch(err => {
            commit("auth_error");
            localStorage.removeItem("token");
            localStorage.removeItem("username");
            localStorage.removeItem("level");
            localStorage.removeItem("points");
            reject(err);
          });
      });
    },
    logout({ commit }) {
      return new Promise(resolve => {
        commit("logout");
        localStorage.removeItem("token");
        localStorage.removeItem("username");
        localStorage.removeItem("level");
        localStorage.removeItem("points");
        delete axios.defaults.headers.common["Authorization"];
        resolve();
      });
    },
    update_level_points({ commit }, level_points_new) {
      return new Promise(resolve => {
        const level_new = level_points_new.level;
        const points_new = level_points_new.points;
        commit("update_level_points", { level_new, points_new });
        resolve();
      });
    }
  },
  getters: {
    isLoggedIn: state => !!state.token,
    isAdmin: state => ["zach", "sherry", "hollyr","alice"].includes(state.username),
    status: state => state.status,
    username: state => state.username,
    token: state => state.token,
    level: state => state.level,
    points: state => state.points
  }
});
