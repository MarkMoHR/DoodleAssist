<template>
    <v-container fluid :fill-height="!iOS" class="py-16">
        <!-- /tracer overlay -->

        <v-row justify-center dense>
            <v-spacer />
            <v-col cols="2" >
                <!-- <p style="line-height: 4;">&nbsp;</p> -->
                <v-card class="mt-16 overflow-auto" style="height:75vh">
                    <v-tabs v-model="tab" background-color="primary" centered dark fixed-tabs>
                        <v-tab>Toolbar</v-tab>
                    </v-tabs>

                    <v-tabs-items v-model="tab">
                        <v-tab-item class="pa-6">
                            <h3>
                                Prompt
                            </h3>
                            <h4>
                                {{ prompt }}
                            </h4>
                            <v-text-field class="my-4" v-model="customizedPrompt"
                                label="Customized Prompt"></v-text-field>
                            <div class="mt-2">
                                <h4>Quick Prompts</h4>
                                <v-chip-group multiple column active-class="primary" v-model="quickSelection">
                                    <v-chip v-for="quickPrompt in quickPrompts" :key="quickPrompt">
                                        {{ quickPrompt }}
                                    </v-chip>
                                </v-chip-group>
                            </div>


                            <v-expansion-panels class="mt-4" accordion>
                                <v-expansion-panel>
                                    <v-expansion-panel-header>
                                        <div>
                                            <v-icon left>mdi-arm-flex-outline</v-icon> Action Prompt
                                        </div>
                                    </v-expansion-panel-header>
                                    <v-expansion-panel-content>
                                        <v-chip-group multiple column active-class="primary" v-model="actionSelection">
                                            <v-chip v-for="actionPrompt in actionPrompts" :key="actionPrompt">
                                                {{ actionPrompt }}
                                            </v-chip>
                                        </v-chip-group>
                                    </v-expansion-panel-content>
                                </v-expansion-panel>

                                <v-expansion-panel>
                                    <v-expansion-panel-header>
                                        <div><v-icon left> mdi-face-woman</v-icon> Hair Prompt</div>
                                    </v-expansion-panel-header>
                                    <v-expansion-panel-content>
                                        <v-chip-group multiple column active-class="primary" v-model="hairSelection">
                                            <v-chip v-for="hairPrompt in hairPrompts" :key="hairPrompt">
                                                {{ hairPrompt }}
                                            </v-chip>
                                        </v-chip-group>
                                    </v-expansion-panel-content>
                                </v-expansion-panel>


                                <v-expansion-panel>
                                    <v-expansion-panel-header>
                                        <div><v-icon left>mdi-tshirt-crew-outline</v-icon> Dress Prompt</div>
                                    </v-expansion-panel-header>
                                    <v-expansion-panel-content>

                                        <v-chip-group multiple column active-class="primary" v-model="dressSelection">
                                            <v-chip v-for="dressPrompt in dressPrompts" :key="dressPrompt">
                                                {{ dressPrompt }}
                                            </v-chip>
                                        </v-chip-group>
                                    </v-expansion-panel-content>
                                </v-expansion-panel>


                                <v-expansion-panel>
                                    <v-expansion-panel-header>
                                        <div><v-icon left>mdi-hat-fedora</v-icon> Decoration Prompt</div>
                                    </v-expansion-panel-header>
                                    <v-expansion-panel-content>

                                        <v-chip-group multiple column active-class="primary"
                                            v-model="decorationSelection">
                                            <v-chip v-for="decorationPrompt in decorationPrompts"
                                                :key="decorationPrompt">
                                                {{ decorationPrompt }}
                                            </v-chip>
                                        </v-chip-group>
                                    </v-expansion-panel-content>
                                </v-expansion-panel>


                            </v-expansion-panels>
                            <div class="mt-8">
                                <h4>Mask Size</h4>
                                <v-slider class="mt-2" dense thumb-label v-model="sizeMask" min="15" max="100"
                                    append-icon="mdi-plus-circle" prepend-icon="mdi-minus-circle"></v-slider>
                            </div>

                            <div>
                                <h4>Visibility</h4>
                                <div class="d-flex justify-space-between mt-1">
                                    <v-btn  :color="visStroke ? 'primary' : 'secondary'"
                                        @click="visStroke = !visStroke"> Stroke
                                        <v-icon right v-if="visStroke">mdi-eye</v-icon>
                                        <v-icon right v-if="!visStroke">mdi-eye-off</v-icon>
                                    </v-btn>

                                    <v-btn  :color="visMask ? 'primary' : 'secondary'"
                                        @click="visMask = !visMask"> Mask
                                        <v-icon right v-if="visMask">mdi-eye</v-icon>
                                        <v-icon right v-if="!visMask">mdi-eye-off</v-icon>
                                    </v-btn>

                                    <v-btn  :color="visImage ? 'primary' : 'secondary'"
                                        @click="visImage = !visImage"> Line Art
                                        <v-icon right v-if="visImage">mdi-eye</v-icon>
                                        <v-icon right v-if="!visImage">mdi-eye-off</v-icon>
                                    </v-btn>
                                </div>

                            </div>
                        </v-tab-item>
                    </v-tabs-items>

                </v-card>
            </v-col>

            <v-col cols="4" align-self="center">
                <v-row justify-center class="mb-2">
                    <div class="d-flex justify-space-around mx-auto">
                        <div class="mx-8">
                            <div class="text-center my-2 text-h6">
                                Stroke

                            </div>
                            <div class="d-flex justify-center flex-wrap">
                                <v-btn class="mx-2" :color="(mode == 'stroke_draw') ? 'primary' : 'secondary'"
                                    @click="mode = 'stroke_draw'"><v-icon
                                        left>mdi-pencil</v-icon>Draw</v-btn>
                                <v-btn class="mx-2" :color="(mode == 'stroke_select') ? 'primary' : 'secondary'"
                                    @click="mode = 'stroke_select'" :disabled="strokes.length == 0
                                        "><v-icon left>mdi-cursor-default</v-icon>Edit</v-btn>
                                <v-btn class="mx-2" :color="(mode == 'stroke_erase') ? 'primary' : 'secondary'"
                                    @click="mode = 'stroke_erase'" :disabled="strokes.length == 0
                                        "><v-icon left> mdi-eraser</v-icon>Delete</v-btn>
                            </div>
                        </div>

                        <div class="mx-8">
                            <div class="text-center my-2 text-h6">Mask

                            </div>
                            <div class="d-flex justify-center flex-wrap">
                                <v-btn class="mx-2" :color="(mode == 'mask_draw') ? 'primary' : 'secondary'"
                                    @click="mode = 'mask_draw'"><v-icon
                                        left>mdi-checkbox-blank-circle</v-icon>Draw</v-btn>
                                <v-btn class="mx-2" :color="(mode == 'mask_erase') ? 'primary' : 'secondary'"
                                    @click="mode = 'mask_erase'"><v-icon
                                        left>mdi-transition-masked</v-icon>Erase</v-btn>
                            </div>
                        </div>
                    </div>
                </v-row>

                <v-row>
                    <v-card class="elevation-12 mx-auto" :width="card_width" :height="card_height">
                        <div ref="canvas" :style="{
                            cursor: mode !== 'stroke_select' ? 'crosshair' : 'default',
                            position: 'absolute',
                            top: '0',
                            left: '0',
                            width: '100%',
                            height: '100%',
                            zIndex: 2
                        }">

                        </div>
                        <canvas ref="resultCanvas" :width="card_width" :height="card_height"
                            style="position: absolute;top: 0;left: 0;pointer-events: none; z-index: 1;"></canvas>

                        <canvas ref="maskCanvas" :width="card_width" :height="card_height"
                            style="display:none;"></canvas>
                        <canvas ref="backgroundCanvas" :width="card_width" :height="card_height"
                            style="display:none;"></canvas>

                    </v-card>
                </v-row>

                <div class="mt-16">

                    <div class="d-flex justify-center mx-auto">
                        <!-- <v-btn class="mx-auto my-4" color="primary" @click="suggest">Suggest</v-btn> -->
                        <v-btn class="mx-auto" color="primary" @click="generate">Generate</v-btn>
                        <!-- <v-btn class="mx-auto my-4" color="primary" @click="selectResult()">Select</v-btn> -->
                    </div>

                </div>
            </v-col>

            <v-col cols="4" align-self="center" fill-height>
                <v-card align-self="center" :width="card_width" :height="card_height" elevation="0"
                    class="d-flex justify-center align-self-center flex-wrap pa-0">
                    <v-row>
                        <v-col v-for="(item, i) in items" :key="i" class="d-flex child-flex" cols="6">
                            <v-card @click="selectResult(i)" elevation="9" :disabled="item.src == ''"
                                :loading="isGenerating" style="aspect-ratio: 1;">
                                <img :src="item.src" style="aspect-ratio: 1; max-width: 100%; display: block;" />
                            </v-card>
                        </v-col>
                    </v-row>

                </v-card>
            </v-col>
            <v-spacer />

        </v-row>

        <v-snackbar v-model="snackbar" :color="snackbar_color" :timeout="2000">
            {{ snackbar_text }}
        </v-snackbar>
        <v-snackbar v-model="snackbar_orientation" color="error" :timeout="-1">
            Please use landscape mode to proceed
        </v-snackbar>
        <v-dialog v-model="dialog" width="500">
            <v-card>
                <v-card-title class="headline warning">Warning</v-card-title>
                <v-card-text>Sketch has not been saved. Are you sure to leave this page?
                </v-card-text>
                <v-card-actions>
                    <v-spacer />
                    <v-btn color="error" text @click="
                        is_leaving = true;
                    $router.push(to_path);
                    ">Leave</v-btn>
                    <v-btn color="primary" text @click="dialog = false">Stay</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
    </v-container>
</template>

<script>
import axios from "axios";
import Raphael from "raphael";
// import StrokeAttr from "./StrokeAttr";
import rotateSVG from "@/assets/rotate.svg";

export default {
    name: "Tracer",
    components: {
        // "sketch-picker": VueColor.Sketch,
        // StrokeAttr: StrokeAttr
    },
    data() {
        return {
            sizeMask: 30,

            visMask: true,
            visStroke: true,
            visImage: true,

            deletedPaths: [],
            selectedPaths: [],
            selectedSet: null,
            selectorGrip: null,
            rotationGrip: null,

            backendAPI: [{ id: 0, url: "http://localhost:9000" }, { id: 1, url: "http://localhost:9001" }, { id: 2, url: "http://localhost:9002" }, { id: 3, url: "http://localhost:9003" }],

            isGenerating: false,
            customizedPrompt: "",
            // genderPrompts: ["1boy", "1girl"],
            // genderSelection: [],
            quickPrompts: [
                "1girl",
                "1boy",
                "close-up",
                "smile",
                "long hair",
                "short hair",
                "open mouth",
                "closed mouth",
                "shirt",
                "dress",
                "skirt",
                "school uniform",
                "cat ears",
                "cat tail",
                "fox tail",
                "long sleeves",
                "short sleeves",
                "bowtie",
                "hat",
                "ponytail",
            ],
            quickSelection: [0],
            actionPrompts: ["outstretched arm",
                "outstretched hand",
                "pointing",
                "pointing at viewer",
                "reaching out",],
            actionSelection: [],
            hairPrompts: ["very long hair",
                "bangs",
                "swept bangs",
                "blunt bangs",
                "braid",
                "single braid",
                "wavy hair",
                "hair between eyes",
                "hair over one eye",
                "hair ornament",
                "hair bow",
                "hair ribbon",],
            hairSelection: [],
            dressPrompts: [
                "collared shirt",
                "jacket",
                "apron",
                "pants",
                "shorts",
                "pleated skirt",
                "maid",
                "hoodie",
                "belt",
                "vest",
                "shoes",
                "puffy sleeves",
                "wide sleeves",
                "sleeveless",],
            dressSelection: [],
            decorationPrompts: ["bow",
                "ribbon",
                "jewelry",
                "necktie",
                "necklace",
                "choker",
                "earrings",
                "scarf",
                "headphones",
                "witch hat",
                "beret",
                "wings",
                "angel wings",
                "flower",
                "balloon",
                "book",
                "holding weapon",
                "sword",],
            decorationSelection: [],
            items: [{ src: '' },
            { src: '' },
            { src: '' },
            { src: '' },
            ],
            selected: [],
            isMaskEdited: false,
            tab: null,

            snackbar: false,
            snackbar_color: "error",
            snackbar_text: "",
            snackbar_nonstylus: false,
            switch_nonstylus: true,
            snackbar_orientation: false,
            dialog: false,
            to_path: "",
            is_leaving: false,
            addbeforeunload: false,
            iOS: false,
            total_id: 0,
            card_width: 0,
            card_height: 0,
            size_ratio: 0,
            image: "empty.png",
            paper: undefined,
            is_drawing: false,
            is_outcanvas: false,
            mode: "stroke_draw",
            strokes: [],
            strokes_boundingbox: [],
            curr_stroke: {
                txy: [],
                p: [],
                color: "",
                opacity: 1,
                width: 3,
                stylus: true
            },
            curr_path: undefined,
            curr_path_string: "",
            // timer: undefined,
            // timer_limit: 120, // const
            // timer_color: "primary",
            batch_size: 15, // const
            colors: { a: 1, hex: "#000000" },
            selectedColors: { a: 1, hex: "#CC0033" },
            stroke_width: 3,
            touch_offsetX: 0,
            touch_offsetY: 0,
            num_strokes_before_update_points: Math.floor(Math.random() * 16 + 1),
            all_points_to_update: [10, 10, 10, 10, 10, 15, 15, 15, 20, 20, 30],
            all_messages: ["Good job!", "Nice work!", "Amazing!", "Keep it up!"],
            prev_level: Number(this.$store.getters.level),
            prev_points: Number(this.$store.getters.points),
            num_tracings: 0,
            next_button_text: "",
            dialog_cantdraw: false,
            tablet_info: "",
            stylus_info: "",
            browser_info: "",
            required_rules: [v => !!v || "This is required"]
        };
    },
    computed: {
        prompt: function () {
            let result = [];
            // result.push(this.genderPrompts[this.genderSelection]);
            this.quickSelection.forEach(_ => { result.push(this.quickPrompts[_]) });
            this.actionSelection.forEach(_ => { result.push(this.actionPrompts[_]) });
            this.hairSelection.forEach(_ => { result.push(this.hairPrompts[_]) });
            this.dressSelection.forEach(_ => { result.push(this.dressPrompts[_]) });
            this.decorationSelection.forEach(_ => { result.push(this.decorationPrompts[_]) });

            result = result.join(', ')
            if (this.customizedPrompt != "") {
                if (result != "") {
                    result = result + ', ' + this.customizedPrompt;
                }
                else {
                    result = this.customizedPrompt;
                }
            }
            return result;
        }
    },
    watch: {
        visImage(_, __) { this.redrawImg(); },
        visMask(_, __) { this.redrawImg(); },
        visStroke(_, __) { 
            this.paper.forEach(_ ? e => { e.show(); } : e => { e.hide(); }) 
        },
        tab(newValue, oldValue) {
            if (newValue == 1) {
                this.mode = "mask_draw"
                this.paper.forEach((el) => { el.hide() })
            }
            else {
                this.paper.forEach((el) => { el.show() })
            }
        },
        mode(newValue, oldValue) {
            if (oldValue == "stroke_select") {
                this.paper.forEach((i) => { i.attr({ "cursor": "" }) });
                this.selectedPaths = [];
            }
            if (newValue == "stroke_select") {
                this.paper.forEach((i) => { i.attr({ "cursor": "pointer" }) });
            }
        },
        selectedPaths() {
            this.paper.forEach((path) => { path.attr({ "stroke": this.colors.hex, "stroke-opacity": this.colors.a }) });
            this.selectedPaths.forEach((path) => { path.attr({ "stroke": this.selectedColors.hex, "stroke-opacity": this.selectedColors.a }) });
            this.selectedSet = null;
            if (this.selectorGrip) {
                this.selectorGrip.remove();
                this.selectorGrip = null;
                this.rotationGrip.remove();
                this.rotationGrip = null;
            }
            if (this.selectedPaths.length) {
                this.selectedSet = this.paper.set();
                this.selectedSet.push(...this.selectedPaths);

                let { x, y, width, height } = this.selectedPaths.length == 1 ? this.selectedSet[0].getBBox(true) : this.selectedSet.getBBox();
                
                this.selectorGrip = this.paper.set();
                this.selectorGrip.push(
                    this.paper.circle(x, y, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x + width / 2, y, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x + width, y, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x + width, y + height / 2, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x + width, y + height, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x + width / 2, y + height, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x, y + height, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.circle(x, y + height / 2, 4).attr({ "fill": this.selectedColors.hex }),
                    this.paper.rect(x, y, width, height).attr({ "cursor": "move", "stroke-dasharray": "--", "fill": "#FFFFFF", "fill-opacity": 0 }).toBack()
                );
                this.selectorGrip.attr({ "stroke": this.selectedColors.hex });

                this.rotationGrip = this.paper.circle(this.selectorGrip[1].attr("cx"), this.selectorGrip[1].attr("cy") - 16, 6).attr({ "stroke": "none", "fill": "#3399FF" });
                this.rotationGrip.node.style.cursor = `url(${rotateSVG}) 12 12, auto`

                for (let i = 0; i < 8; ++i) {
                    this.selectorGrip[i].node.setAttribute("vector-effect", "non-scaling-size")
                }
                this.selectorGrip[8].node.setAttribute("vector-effect", "non-scaling-stroke");
                this.rotationGrip.node.setAttribute("vector-effect", "non-scaling-size");


                if (this.selectedPaths.length == 1) {
                    let m = this.selectedPaths[0].matrix.clone();
                    let t = m.split();
                    this.selectorGrip.transform(m.toTransformString());
                    this.rotationGrip.transform(m.toTransformString());
                }

                this.assignCursor();

                // Avoid propagtion to canvas
                this.selectorGrip.forEach((ele) => {
                    ele.node.addEventListener("pointerdown", (e) => {
                        e.stopPropagation();
                    })
                });
                this.rotationGrip.node.addEventListener("pointerdown", (e) => {
                    e.stopPropagation();
                })

                // Translation event listener
                let lastDx = 0, lastDy = 0;
                this.selectorGrip[8].drag((dx, dy) => {
                    let tstr = `...T${dx - lastDx},${dy - lastDy}`;
                    this.selectorGrip.transform(tstr);
                    this.rotationGrip.transform(tstr);
                    this.selectedSet.transform(tstr);

                    lastDx = dx;
                    lastDy = dy;
                }, function (x, y) {
                }, () => {
                    lastDx = 0;
                    lastDy = 0;
                    this.assignCursor();
                    console.log("Translate")
                    this.suggest("edit", this.selectedPaths.map((path) => path.data("id")));
                });

                // Rotation event listener
                let lastDeg = 0, offsetX = this.paper.canvas.getBoundingClientRect().left, offsetY = this.paper.canvas.getBoundingClientRect().top;
                this.rotationGrip.drag((dx, dy, x, y) => {
                    x = x - offsetX;
                    y = y - offsetY;
                    let bbox = this.selectorGrip.getBBox();
                    let sx = x - dx, sy = y - dy;
                    let cx = bbox.x + bbox.width / 2, cy = bbox.y + bbox.height / 2;

                    let deg = this.calculateAngle(cx - sx, cy - sy, cx - x, cy - y);
                    // console.log(deg);
                    // let tstr = `...T${-cx},${-cy}R${deg - lastDeg},0,0T${cx},${cy}`
                    let tstr = `...R${deg - lastDeg},${cx},${cy}`
                    this.selectorGrip.transform(tstr);
                    this.rotationGrip.transform(tstr);
                    this.selectedSet.transform(tstr);
                    lastDeg = deg;

                }, function (x, y, e) {
                }, () => {
                    lastDeg = 0;
                    this.assignCursor();
                    console.log("Rotation")
                    this.suggest("edit", this.selectedPaths.map((path) => path.data("id")));
                });

                //Scale event listener
                //Corner points
                // for (let i of [0, 2, 4, 6]) {
                for (let i = 0; i < 8; ++i) {
                    let cornerGrip = this.selectorGrip[i];
                    cornerGrip.drag((dx, dy, x, y) => {
                        x = x - offsetX;
                        y = y - offsetY;
                        let bbox = this.selectorGrip.getBBox();
                        let sx = cornerGrip.matrix.x(cornerGrip.attrs.cx, cornerGrip.attrs.cy), sy = cornerGrip.matrix.y(cornerGrip.attrs.cx, cornerGrip.attrs.cy);
                        let cx = bbox.x + bbox.width / 2, cy = bbox.y + bbox.height / 2;
                        let ax = x - cx, ay = y - cy;
                        // console.log(cornerGrip,cornerGrip.cx);
                        let bx = sx - cx, by = sy - cy, bLen = Math.sqrt(bx * bx + by * by);
                        // console.log(bx, by);
                        // Length of Projected Vec A on Vec B
                        let l = (ax * bx + ay * by) / bLen;
                        // console.log(l, l / bLen);
                        let tstr = `...S${l / bLen},${l / bLen},${cx},${cy}`;
                        this.selectedSet.transform(tstr);
                        this.selectorGrip.transform(tstr);
                        // this.selectorGrip.attrs()
                        this.rotationGrip.transform(tstr);

                    }, function (x, y) {
                    }, () => {
                        this.assignCursor();
                        console.log("Scale");
                        this.suggest("edit", this.selectedPaths.map((path) => path.data("id")));
                        // lastRatio = 1;
                    });
                }
                
            }
        }
    },
    methods: {
        assignCursor() {
            if (this.selectorGrip) {
                let cursors = ['ns-resize', 'nesw-resize', 'ew-resize', 'nwse-resize', 'ns-resize', 'nesw-resize', 'ew-resize', 'nwse-resize']
                let bbox = this.selectorGrip[0].getBBox();
                let x = bbox.x + bbox.width / 2, y = bbox.y + bbox.height / 2;
                bbox = this.selectorGrip.getBBox();
                let cx = bbox.x + bbox.width / 2, cy = bbox.y + bbox.height / 2;
                let deg = this.calculateAngle(0, 1, cx - x, cy - y);
                let idx = Math.floor((deg + 22.5) / 45) % 8;

                for (let i = 0; i < 8; ++i) {
                    this.selectorGrip[i].node.style.cursor = cursors[(idx + i) % 8]
                }
            }
        },
        calculateAngle(Ax, Ay, Bx, By) {
            // Step 1: Calculate the dot product
            const dotProduct = Ax * Bx + Ay * By;

            // Step 2: Calculate the magnitudes of the vectors
            const magnitudeA = Math.sqrt(Ax * Ax + Ay * Ay);
            const magnitudeB = Math.sqrt(Bx * Bx + By * By);

            // Step 3: Calculate the angle in radians
            let angle = Math.acos(dotProduct / (magnitudeA * magnitudeB));

            // Step 4: Calculate the cross product (2D)
            const crossProduct = Ax * By - Ay * Bx;

            // Step 5: Determine if the angle is clockwise or counterclockwise
            if (crossProduct < 0) {
                // Clockwise, subtract from 360
                angle = 2 * Math.PI - angle;
            }

            // Step 6: Convert the angle from radians to degrees (0 to 360)
            const angleInDegrees = angle * (180 / Math.PI);

            return angleInDegrees;
        },
        round_float(f) {
            return Number(parseFloat(f).toFixed(2));
        },
        redrawImg() {
            const maskCanvas = this.$refs.maskCanvas;
            const backgroundCanvas = this.$refs.backgroundCanvas;
            const canvas = this.$refs.resultCanvas;
            const ctx1 = maskCanvas.getContext('2d');
            const ctx2 = backgroundCanvas.getContext('2d');
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            if (this.visImage) {
                ctx.drawImage(backgroundCanvas, 0, 0);
            }

            ctx.globalCompositeOperation = 'screen';
            ctx.fillStyle = '#0000FF';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.globalCompositeOperation = 'darken';
            if (this.visMask) {
                ctx.drawImage(maskCanvas, 0, 0);
            }
            ctx.globalCompositeOperation = 'lighter';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.globalCompositeOperation = 'source-over';
        },
        redraw() {
            this.$refs.canvas.innerHTML = "";
            this.paper = new Raphael(
                this.$refs.canvas,
                this.card_width,
                this.card_height
            );
            for (let i = 0; i < this.strokes.length; i++) {
                if (this.strokes[i].p.length > 0 && this.strokes[i].opacity == 1) {
                    // strokes with varying pressure and opaque color
                    for (let j = 1; j < this.strokes[i].txy.length / 3; j++) {
                        let path_string =
                            "M" +
                            this.round_float(
                                this.strokes[i].txy[3 * j - 2] * this.size_ratio
                            ) +
                            "," +
                            this.round_float(
                                this.strokes[i].txy[3 * j - 1] * this.size_ratio
                            ) +
                            "L" +
                            this.round_float(
                                this.strokes[i].txy[3 * j + 1] * this.size_ratio
                            ) +
                            "," +
                            this.round_float(
                                this.strokes[i].txy[3 * j + 2] * this.size_ratio
                            );
                        let path = this.paper.path(path_string);
                        path.attr({
                            stroke: this.strokes[i].color,
                            "stroke-linecap": "round",
                            "stroke-width": this.round_float(
                                this.strokes[i].width *
                                this.size_ratio *
                                (this.strokes[i].p[j - 1] + this.strokes[i].p[j])
                            ),
                            "stroke-opacity": this.strokes[i].opacity
                        });
                        // console.log("svg_" + this.strokes[i].id);
                        path.data("id", this.strokes[i].id).node.setAttribute("id", "svg_" + this.strokes[i].id);
                    }
                } else {
                    // strokes with no available pressure or translucent color
                    let pressure_tapering = 1;
                    if (this.strokes[i].p.length > 0) {
                        let average_pressure = 0;
                        for (let j = 0; j < this.strokes[i].p.length; j++) {
                            average_pressure += this.strokes[i].p[j];
                        }
                        average_pressure /= this.strokes[i].p.length;
                        pressure_tapering = average_pressure * 2;
                    }
                    let path_string =
                        "M" +
                        this.round_float(this.strokes[i].txy[1] * this.size_ratio) +
                        "," +
                        this.round_float(this.strokes[i].txy[2] * this.size_ratio);
                    for (let j = 1; j < this.strokes[i].txy.length / 3; j++) {
                        path_string +=
                            "L" +
                            this.round_float(
                                this.strokes[i].txy[3 * j + 1] * this.size_ratio
                            ) +
                            "," +
                            this.round_float(
                                this.strokes[i].txy[3 * j + 2] * this.size_ratio
                            );
                    }
                    let path = this.paper.path(path_string);
                    path.attr({
                        stroke: this.strokes[i].color,
                        "stroke-linecap": "round",
                        "stroke-linejoin": "round",
                        "stroke-width": this.round_float(
                            this.strokes[i].width * this.size_ratio * pressure_tapering
                        ),
                        "stroke-opacity": this.strokes[i].opacity
                    });
                    // console.log("svg_" + this.strokes[i].id);
                    path.node.setAttribute("id", "svg_" + this.strokes[i].id);
                }
            }
        },
        save_stroke() {
            if (this.curr_stroke.txy.length > 0) {
                this.curr_stroke.color = this.colors.hex;
                this.curr_stroke.opacity = this.colors.a;
                this.curr_stroke.width = this.stroke_width;
                this.strokes.push(this.curr_stroke);
                // save bounding box for erasing
                let minx = 800,
                    miny = 800,
                    maxx = 0,
                    maxy = 0;
                for (let i = 0; i < this.curr_stroke.txy.length; i += 3) {
                    minx =
                        this.curr_stroke.txy[i + 1] < minx
                            ? this.curr_stroke.txy[i + 1]
                            : minx;
                    miny =
                        this.curr_stroke.txy[i + 2] < miny
                            ? this.curr_stroke.txy[i + 2]
                            : miny;
                    maxx =
                        this.curr_stroke.txy[i + 1] > maxx
                            ? this.curr_stroke.txy[i + 1]
                            : maxx;
                    maxy =
                        this.curr_stroke.txy[i + 2] > maxy
                            ? this.curr_stroke.txy[i + 2]
                            : maxy;
                }
                this.strokes_boundingbox.push({
                    minx: minx,
                    miny: miny,
                    maxx: maxx,
                    maxy: maxy
                });
            }
            this.curr_stroke = {
                txy: [],
                p: [],
                color: "",
                opacity: 1,
                width: 3,
                stylus: true
            };
            if (!this.visStroke) {
                this.curr_path.hide();
            }
            
        },
        pointerdown(e) {
            // return if not capturing non-stylus data
            if (!this.switch_nonstylus) {
                if (
                    !(
                        e.pointerType == "pen" ||
                        (e.type.substring(0, 5) == "touch" &&
                            e.touches[0].touchType == "stylus")
                    )
                ) {
                    return;
                }
            }

            this.is_drawing = true;
            this.is_outcanvas = false;
            let x = this.round_float(e.offsetX),
                y = this.round_float(e.offsetY);
            if (e.type.substring(0, 5) == "touch") {
                if (this.touch_offsetX == 0 && this.touch_offsetY == 0) {
                    let rect = e.target.getBoundingClientRect();
                    this.touch_offsetX = rect.left;
                    this.touch_offsetY = rect.top;
                }
                x = this.round_float(e.touches[0].clientX - this.touch_offsetX);
                y = this.round_float(e.touches[0].clientY - this.touch_offsetY);
            }
            if (this.mode == "stroke_erase") {
                let el = this.paper.getElementByPoint(e.x, e.y);
                if (el != null) {

                    this.deletedPaths.push(el.data("id"));
                    el.remove();
                }
                return;
            }
            if (this.mode == "stroke_draw") {
                this.total_id = this.total_id + 1;
                this.curr_stroke.id = this.total_id;
                // console.log(this.total_id)
                this.curr_stroke.txy.push(
                    Date.now(),
                    this.round_float(x / this.size_ratio),
                    this.round_float(y / this.size_ratio)
                );
                if (e.pointerType == "pen") {
                    this.curr_stroke.p.push(e.pressure);
                } else if (
                    e.type.substring(0, 5) == "touch" &&
                    e.touches[0].touchType == "stylus"
                ) {
                    this.curr_stroke.p.push(e.touches[0].force);
                } else {
                    // sketches not drawn using stylus will not be saved
                    this.curr_stroke.stylus = false;
                    this.snackbar_nonstylus = true;
                }
                if (this.curr_stroke.p.length == 0 || this.colors.a < 1) {
                    // initialize path for the whole stroke if no pressure data is available or stroke color is translucent
                    this.curr_path_string = "M" + x + "," + y;
                    this.curr_path = this.paper.path(this.curr_path_string);
                    this.curr_path.attr({
                        stroke: this.colors.hex,
                        "stroke-linecap": "round",
                        "stroke-linejoin": "round",
                        "stroke-width": this.round_float(
                            this.stroke_width * this.size_ratio
                        ),
                        "stroke-opacity": this.colors.a
                    });
                    // console.log("svg_" + this.curr_stroke.id);
                    this.curr_path.data("id", this.curr_stroke.id);
                    this.curr_path.node.setAttribute("id", "svg_" + this.curr_stroke.id);
                    // this.curr_path.node.style.cursor = 'pointer';
                    let path = this.curr_path;
                    this.curr_path.node.addEventListener("pointerdown", (e) => {
                        // console.log(this, e);
                        if (this.mode != 'stroke_select') { return; }
                        if (e.ctrlKey) {
                            let idx = this.selectedPaths.indexOf(path);
                            if (idx == -1) {
                                this.selectedPaths.push(path);
                            }
                            else {
                                // this.selectedPaths = this.selectedPaths.filter(item => item !== path)
                                this.selectedPaths.splice(idx, 1);
                            }
                        }
                        else { this.selectedPaths = [path]; }
                        // console.log(path, this.selectedPaths);
                        e.stopPropagation();
                    })
                }
                // warn user before closing/refreshing the page if sketch has not been saved
                if (!this.addbeforeunload) {
                    window.addEventListener("beforeunload", this.warnbeforeunload);
                    window.addEventListener(
                        this.iOS ? "pagehide" : "unload",
                        this.restorepointsunload
                    );
                    this.addbeforeunload = true;
                }
            }
            if (this.mode == "mask_draw") {
                // console.log(x,y);
                this.isMaskEdited = true;
                const maskCanvas = this.$refs.maskCanvas;
                const ctx = maskCanvas.getContext('2d');
                ctx.fillStyle = "rgb(122,122, 122)"; // Fully opaque red
                ctx.beginPath();
                ctx.arc(x, y, this.sizeMask, 0, 2 * Math.PI); // (x, y, radius, startAngle, endAngle)
                ctx.fill(); // Fill the circle with the color
                this.redrawImg();
            }
            if (this.mode == "mask_erase") {
                this.isMaskEdited = true;
                const maskCanvas = this.$refs.maskCanvas;
                const ctx = maskCanvas.getContext('2d');
                ctx.fillStyle = "rgb(255,255, 255)"; // Fully opaque red
                ctx.beginPath();
                ctx.arc(x, y, this.sizeMask, 0, 2 * Math.PI); // (x, y, radius, startAngle, endAngle)
                ctx.fill(); // Fill the circle with the color
                this.redrawImg();
            }
            if (this.mode == "stroke_select") {
                if (e.ctrlKey) { return; }
                this.selectedPaths = [];
            }
        },
        pointermove(e) {
            // return if not capturing non-stylus data
            if (!this.switch_nonstylus) {
                if (
                    !(
                        e.pointerType == "pen" ||
                        (e.type.substring(0, 5) == "touch" &&
                            e.touches[0].touchType == "stylus")
                    )
                ) {
                    return;
                }
            }
            if (this.is_drawing) {
                let x = this.round_float(e.offsetX),
                    y = this.round_float(e.offsetY);
                if (e.type == "touchmove") {
                    x = this.round_float(e.touches[0].clientX - this.touch_offsetX);
                    y = this.round_float(e.touches[0].clientY - this.touch_offsetY);
                }
                // pointerleave and pointerenter events are triggered after all pointermove events for surface pen
                // so check if event is out of canvas and return pointerleave or pointerenter as pointer is moving
                if (!this.is_outcanvas) {
                    if (x < 0 || x > this.card_width || y < 0 || y > this.card_height) {
                        return this.pointerleave(e);
                    } else {
                        if (this.mode == "stroke_erase") {
                            let el = this.paper.getElementByPoint(e.x, e.y);
                            if (el != null) {
                                this.deletedPaths.push(el.data("id"))
                                el.remove();
                            }
                            return;
                        }
                        if (this.mode == "stroke_draw") {
                            this.curr_stroke.txy.push(
                                Date.now(),
                                this.round_float(x / this.size_ratio),
                                this.round_float(y / this.size_ratio)
                            );
                            if (e.pointerType == "pen") {
                                this.curr_stroke.p.push(e.pressure);
                            } else if (
                                e.type == "touchmove" &&
                                e.touches[0].touchType == "stylus"
                            ) {
                                this.curr_stroke.p.push(e.touches[0].force);
                            } else {
                                // sketches not drawn using stylus will not be saved
                                this.curr_stroke.stylus = false;
                                this.snackbar_nonstylus = true;
                            }
                            if (this.curr_stroke.p.length > 0 && this.colors.a == 1) {
                                // render a path for every two points because of varying pressure
                                let prev_x = this.round_float(
                                    this.curr_stroke.txy[this.curr_stroke.txy.length - 5] *
                                    this.size_ratio
                                );
                                let prev_y = this.round_float(
                                    this.curr_stroke.txy[this.curr_stroke.txy.length - 4] *
                                    this.size_ratio
                                );
                                let prev_p = this.curr_stroke.p[this.curr_stroke.p.length - 2];
                                let p = this.curr_stroke.p[this.curr_stroke.p.length - 1];
                                this.curr_path_string =
                                    "M" + prev_x + "," + prev_y + "L" + x + "," + y;
                                this.curr_path = this.paper.path(this.curr_path_string);
                                this.curr_path.attr({
                                    stroke: this.colors.hex,
                                    "stroke-linecap": "round",
                                    "stroke-width": this.round_float(
                                        this.stroke_width * this.size_ratio * (prev_p + p)
                                    ),
                                    "stroke-opacity": this.colors.a
                                });
                            } else {
                                // render one path for the whole stroke if no pressure data is available
                                this.curr_path_string += "L" + x + "," + y;
                                this.curr_path.attr("path", this.curr_path_string);
                                // render one path for the whole stroke if pressure data is available but stroke color is translucent
                                if (this.curr_stroke.p.length > 0 && this.colors.a < 1) {
                                    let average_pressure = 0;
                                    for (let i = 0; i < this.curr_stroke.p.length; i++) {
                                        average_pressure += this.curr_stroke.p[i];
                                    }
                                    average_pressure /= this.curr_stroke.p.length;
                                    this.curr_path.attr(
                                        "stroke-width",
                                        this.round_float(
                                            this.stroke_width * this.size_ratio * average_pressure * 2
                                        )
                                    );
                                }
                            }
                        }
                        if (this.mode == "mask_draw") {
                            // console.log(x,y);
                            const maskCanvas = this.$refs.maskCanvas;
                            const ctx = maskCanvas.getContext('2d');
                            ctx.fillStyle = "rgb(122,122, 122)"; // Fully opaque red
                            ctx.beginPath();
                            ctx.arc(x, y, this.sizeMask, 0, 2 * Math.PI); // (x, y, radius, startAngle, endAngle)
                            ctx.fill(); // Fill the circle with the color
                            this.redrawImg();
                        }
                        if (this.mode == "mask_erase") {
                            const maskCanvas = this.$refs.maskCanvas;
                            const ctx = maskCanvas.getContext('2d');
                            ctx.fillStyle = "rgb(255,255, 255)"; // Fully opaque red
                            ctx.beginPath();
                            ctx.arc(x, y, this.sizeMask, 0, 2 * Math.PI); // (x, y, radius, startAngle, endAngle)
                            ctx.fill(); // Fill the circle with the color
                            this.redrawImg();
                        }
                    }
                } else {
                    if (
                        !(x < 0 || x > this.card_width || y < 0 || y > this.card_height)
                    ) {
                        return this.pointerenter(e);
                    }
                }
            }
        },
        pointerup(e) {
            // return if not capturing non-stylus data
            if (!this.switch_nonstylus) {
                if (!(e.pointerType == "pen" || e.type.substring(0, 5) == "touch")) {
                    return;
                }
            }
            if (this.is_drawing) {
                this.is_drawing = false;
                if (this.mode == "stroke_draw") {
                    if (!this.is_outcanvas) {

                        this.save_stroke();
                    }
                    console.log("Draw")
                    this.suggest("add", [this.curr_path.data("id")]);
                }
                if (this.mode == "stroke_erase") {
                    if (this.deletedPaths.length != 0) {
                        this.suggest("delete", this.deletedPaths);
                        this.deletedPaths = [];
                    }
                }
            }
        },
        pointerleave(e) {
            // return if not capturing non-stylus data
            if (!this.switch_nonstylus) {
                if (
                    !(
                        e.pointerType == "pen" ||
                        (e.type.substring(0, 5) == "touch" &&
                            e.touches[0].touchType == "stylus")
                    )
                ) {
                    return;
                }
            }
            if (this.is_drawing) {
                this.is_outcanvas = true;
                if (this.mode == "stroke_draw") {
                    this.save_stroke();
                }
            }
        },
        pointerenter(e) {
            // return if not capturing non-stylus data
            if (!this.switch_nonstylus) {
                if (
                    !(
                        e.pointerType == "pen" ||
                        (e.type.substring(0, 5) == "touch" &&
                            e.touches[0].touchType == "stylus")
                    )
                ) {
                    return;
                }
            }
            if (this.is_drawing) {
                return this.pointerdown(e);
            }
        },
        generate() {
            this.isGenerating = true;
            this.items.forEach((item) => { item.src = '' });
            let inputDataUpdate = {}, inputData = {};
            inputData.prompt = this.prompt;
            inputData.mode_source = "Novice mode";
            inputDataUpdate.mode_source = "Novice mode";
            if (this.isMaskEdited) {
                inputDataUpdate.mask_url = this.$refs.maskCanvas.toDataURL("image/png");
            }
            else { inputDataUpdate.mask_url = null; }

            const httpCfg = { headers: { "Content-Type": "application/json" } };
            const httpData = JSON.parse(JSON.stringify(inputData));
            const httpDataUpdate = JSON.parse(JSON.stringify(inputDataUpdate));
            axios.post(this.backendAPI[0].url + '/update_mask', httpDataUpdate).then((response) => {
                const requests = this.backendAPI.map((server) =>
                    axios.post(`${server.url}/generate`, httpData, httpCfg)
                        .then((response) => ({
                            server,
                            data: response.data,
                            success: true
                        }))
                        .catch((error) => ({
                            server,
                            error: error.message,
                            success: false
                        }))
                );

                Promise.all(requests)
                    .then((results) => {
                        results.forEach((result) => {
                            if (result.success) {
                                this.items[result.server.id].src = result.data.result_gallery[0];
                            } else {
                                console.error(`Error from ${result.server}:`, result.error);
                            }
                        });
                        this.isGenerating = false; // Update after all requests are processed
                    })
                    .catch((error) => {
                        console.error("Unexpected error:", error);
                    });
            })
                .catch((e) => {
                    alert(e);
                });
        },
        suggest(type, ids) {

            if (this.selectedSet) {
                this.selectedSet.attr({ "stroke": this.colors.hex });
            }

            const svgElement = this.$refs.canvas.querySelector('svg');
            // Get all path elements inside the original svgElement
            const paths = svgElement.querySelectorAll('path');

            let pathString = ""
            paths.forEach(path => {
                pathString += path.outerHTML;
            });

            if (this.selectedSet) {
                this.selectedSet.attr({ "stroke": this.selectedColors.hex });
            }
            // Create a string to hold the filtered SVG content
            let svgString = svgElement.outerHTML.replace(svgElement.innerHTML, pathString);

            let inputData = {};
            inputData.svg = svgString;
            inputData.modified_path_ids = { "add": [], "delete": [], "edit": [] };
            inputData.modified_path_ids[`${type}`] = ids.map((id) => `svg_${id}`);
            console.log(inputData);
            const httpCfg = { headers: { "Content-Type": "application/json" } };
            const httpData = JSON.parse(JSON.stringify(inputData));

            axios.post(this.backendAPI[0].url + "/modify_svg", httpData, httpCfg)
                .then((response) => {
                    const canvas = this.$refs.maskCanvas;
                    const ctx = canvas.getContext('2d');

                    const img = new Image();
                    img.onload = () => {
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        this.redrawImg();
                    };
                    img.src = response.data.mask_url;
                    this.isMaskEdited = false;
                })
                .catch((e) => {
                    alert(e);
                });
        },

        warnbeforeunload(e) {
            e.preventDefault();
            e.returnValue = "";
        },
        restorepointsunload() {
            this.$store.dispatch("update_level_points", {
                level: this.prev_level,
                points: this.prev_points
            });
        },
        preventtouchdefault(e) {
            e.preventDefault();
            if (e.type == "touchstart" && e.target.click) {
                e.target.click();
            }
        },
        orientationchange() {
            if (!(screen.orientation === 90 || screen.orientation === -90)) {
                this.snackbar_orientation = false;
                this.$router.go();
            } else {
                this.snackbar_orientation = true;
            }
        },
        selectResult(i) {
            let inputData = { selected_idx: 0 };
            const httpCfg = { headers: { "Content-Type": "application/json" } };
            const httpData = JSON.parse(JSON.stringify(inputData));

            axios.post(this.backendAPI[i].url + "/select_generation", httpData, httpCfg)
                .then((response) => {
                    const backgroundCanvas = this.$refs.backgroundCanvas;
                    const ctx = backgroundCanvas.getContext('2d');

                    // Create an image object
                    const img = new Image();

                    // Set up the onload event listener
                    img.onload = () => {
                        // Draw the image to the canvas once it's loaded
                        console.log(backgroundCanvas.width, backgroundCanvas.height);
                        ctx.drawImage(img, 0, 0, backgroundCanvas.width, backgroundCanvas.height);
                        this.redrawImg();
                    };
                    img.src = this.items[i].src;

                    this.items.forEach((item, idx, array) => array[idx].src = '');
                    const canvas = this.$refs.maskCanvas;
                    const ctx2 = canvas.getContext('2d');
                    ctx2.fillStyle = 'rgb(255,255,255)';
                    ctx2.fillRect(0, 0, canvas.width, canvas.height);
                    this.isMaskEdited = false;
                    this.redrawImg();
                })
                .catch((e) => {
                    alert(e);
                });
        }
    },
    created() {
        if (this.$store.getters.username == "zach") {
            this.switch_nonstylus = true;
        }
        this.iOS = navigator.platform == "MacIntel";
        // !!navigator.platform && /iPad|iPhone|iPod/.test(navigator.platform);
        if (this.iOS) {
            this.snackbar_orientation = (
                screen.orientation === 90 || screen.orientation === -90
            );
        }
        if (this.$route.meta.overlay) {
            this.card_height = Math.round(window.innerHeight * 0.8);
            this.card_width = this.card_height; // overlay
        } else {
            this.card_height = Math.round(window.innerHeight * 0.6);
            this.card_width = this.card_height; // side by side
        }
    },
    mounted() {
        this.size_ratio = this.card_width / 800;
        this.paper = new Raphael(
            this.$refs.canvas,
            this.card_width,
            this.card_height
        );
        // this is important for preventing touch scrolling and ensuring pointer events not to stop abruptly
        document.addEventListener("touchmove", this.preventtouchdefault, {
            passive: false
        });
        document.addEventListener("touchstart", this.preventtouchdefault, {
            passive: false
        });
        if (this.iOS) {
            window.addEventListener("orientationchange", this.orientationchange);
        }
        // Safari does not support pointer events
        if (
            navigator.userAgent.search("Safari") >= 0 &&
            navigator.userAgent.search("Chrome") < 0
        ) {
            console.log("Safari detected");
            this.$refs.canvas.addEventListener("mousedown", this.pointerdown);
            this.$refs.canvas.addEventListener("mousemove", this.pointermove);
            document.addEventListener("mouseup", this.pointerup);
            this.$refs.canvas.addEventListener("mouseleave", this.pointerleave);
            this.$refs.canvas.addEventListener("mouseenter", this.pointerenter);
            // this is to make sure touchmove events are not interrupted when innerHTML resets
            // see https://stackoverflow.com/questions/33298828/touch-move-event-dont-fire-after-touch-start-target-is-removed
            this.$refs.canvas.addEventListener("touchstart", e => {
                const ontouchend = () => {
                    e.target.removeEventListener("touchmove", this.pointermove);
                    e.target.removeEventListener("touchleave", this.pointerleave);
                    e.target.removeEventListener("touchenter", this.pointerenter);
                    e.target.removeEventListener("touchend", ontouchend);
                    this.pointerup(e);
                };
                e.target.addEventListener("touchmove", this.pointermove);
                e.target.addEventListener("touchleave", this.pointerleave);
                e.target.addEventListener("touchenter", this.pointerenter);
                e.target.addEventListener("touchend", ontouchend);
                this.pointerdown(e);
            });
        } else {
            this.$refs.canvas.addEventListener("pointerdown", this.pointerdown);
            this.$refs.canvas.addEventListener("pointermove", this.pointermove);
            document.addEventListener("pointerup", this.pointerup);
            this.$refs.canvas.addEventListener("pointerleave", this.pointerleave);
            this.$refs.canvas.addEventListener("pointerenter", this.pointerenter);
        }
        // disable right click menu on canvas
        this.$refs.canvas.addEventListener("contextmenu", e => e.preventDefault());


    },
    // warn user before leaving the page if sketch has not been saved
    beforeRouteLeave(to, from, next) {
        if (
            this.strokes.length == 0 ||
            this.is_leaving ||
            !this.$store.getters.isLoggedIn
        ) {
            // remove warning listener in other pages
            if (this.addbeforeunload) {
                window.removeEventListener("beforeunload", this.warnbeforeunload);
                window.removeEventListener(
                    this.iOS ? "pagehide" : "unload",
                    this.restorepointsunload
                );
                this.addbeforeunload = false;
            }
            document.removeEventListener("touchstart", this.preventtouchdefault, {
                passive: false
            });
            document.removeEventListener("touchmove", this.preventtouchdefault, {
                passive: false
            });
            if (this.iOS) {
                window.removeEventListener("orientationchange", this.orientationchange);
            }
            this.$store.dispatch("update_level_points", {
                level: this.prev_level,
                points: this.prev_points
            });
            next();
        } else {
            this.to_path = to.path;
            this.dialog = true;
        }
    }
};
</script>