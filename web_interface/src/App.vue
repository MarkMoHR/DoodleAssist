<template>
  <v-app>
    <v-app-bar
      app
      dark
      color="primary"
      v-if="!['subset', 'synthesis'].includes($route.name)"
    >
      <v-toolbar-title class="headline">
        <span
          @click="$router.push('/')"
          @touchstart="$router.push('/')"
          style="cursor: pointer"
          >DoodleAssist</span
        >
      </v-toolbar-title>
      <v-spacer />
      <transition name="fade" mode="out-in"
        ><v-btn
          text
          v-if="$store.getters.isLoggedIn"
          class="text-none"
          style="pointer-events: none"
          >Welcome {{ $store.getters.username }}</v-btn
        ></transition
      >
      <transition name="fade" mode="out-in"
        ><v-btn
          text
          v-if="$store.getters.isLoggedIn"
          class="text-none"
          style="pointer-events: none"
          >Level: {{ $store.getters.level }}</v-btn
        ></transition
      >
      <transition name="fade" mode="out-in"
        ><v-btn
          text
          v-if="$store.getters.isLoggedIn"
          class="text-none"
          style="pointer-events: none"
          >Points: {{ $store.getters.points }}</v-btn
        ></transition
      >
      <transition name="fade" mode="out-in"
        ><v-btn
          text
          v-if="$store.getters.isLoggedIn"
          @click="$router.push('/update')"
          >Update Info
        </v-btn></transition
      >
      <transition name="fade" mode="out-in"
        ><v-btn text v-if="$store.getters.isLoggedIn" @click="logout"
          >Logout</v-btn
        ></transition
      >
    </v-app-bar>
    <v-content>
      <transition name="fade" mode="out-in"><router-view /></transition>
    </v-content>
    <v-snackbar v-model="snackbar" :color="snackbar_color" :timeout="2000">
      {{ snackbar_text }}
    </v-snackbar>
    <v-dialog v-model="dialog" width="500">
      <v-card>
        <v-card-title class="headline warning">Warning</v-card-title>
        <v-card-text
          >Stroke data has not been saved. Are you sure to leave this
          page?</v-card-text
        >
        <v-card-actions>
          <v-spacer />
          <v-btn
            color="error"
            text
            @click="
              is_leaving = true;
              logout();
            "
            >Leave</v-btn
          >
          <v-btn color="primary" text @click="dialog = false">Stay</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-app>
</template>

<script>
export default {
  name: "App",
  data() {
    return {
      snackbar: false,
      snackbar_color: "",
      snackbar_text: "",
      dialog: false,
      is_leaving: false
    };
  },
  methods: {
    logout() {
      if (this.$route.name == "tracer") {
        this.dialog = true;
      }
      if (this.$route.name != "tracer" || this.is_leaving) {
        this.dialog = false;
        this.is_leaving = false;
        this.$store
          .dispatch("logout")
          .then(() => {
            this.snackbar = true;
            this.snackbar_color = "success";
            this.snackbar_text = "Logout successful";
            setTimeout(() => this.$router.push("/login"), 2200);
          })
          .catch(err => console.log(err));
      }
    }
  }
};
</script>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}
html,
body {
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
</style>
