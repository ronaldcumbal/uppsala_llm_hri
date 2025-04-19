const success_url = 'https://app.prolific.com/submissions/complete?cc=CFB2FPWW'

var jsPsych = initJsPsych({
    show_progress_bar: true,
    override_safe_mode: true,
    on_finish: function () {
        window.location.href = success_url;
    }
})

// ----->>>>> Paste code below this line from test HTML

var canvas_width =  1024
var canvas_height = 768
var canvas_border_width = 1
var stroke_width = 1
var stroke_color =  'blue'
var stroke_color_palette = ['blue', 'red', 'yellow', 'magenta', 'green', 'cyan']
var sketch_duration = 180000
var icon_width = 120
var icon_height = 100
var required_flag = true
var timeline = [];

var preload = {
    type: jsPsychPreload,
    images: ['images/components.png',
        'images/icons_text.png',
        'images/icons_symbol.png',
        'images/icons_human.png',
        'images/icons_led.png',
        'images/icons_projection.png',
        'images/icons_sounds.png',
        'images/icons_speech.png',
        'https://ronaldcumbal.github.io/open_repository/images/test.png',
        'https://ronaldcumbal.github.io/open_repository/images/sketch.png',
        'https://ronaldcumbal.github.io/open_repository/images/sketch_high.png',
      ]
  };
  timeline.push(preload);

  
/* ============================= End ============================= */
// Exit Survey
// Note: This survey comes after the last task with important data for my study and provides some buffer time
// at the end of the experiment for the data to finish uploading to the server.

var exit_survey = {
    type: jsPsychSurveyHtmlForm,
    button_label: 'Continue',
    html: 
        '<style>p {text-align:left; spellcheck=false;} input[type="text"] {width:8ch;} fieldset {border:1px solid #999;box-shadow:2px 2px 5px #999;} legend {background:#fff;text-align:left;font-size:110%;}</style>'+
        '<span style="font-size:125%; font-weight:bold;">Exit Survey</span>'+
        '<p>This was a test-version of the survey. Do you have any feedback on the tasks?</p>'+
        '<p style="text-align:left;"><textarea name="feedback" rows="3" style="width:90%;" required></textarea></p>'+
        '<p>If you have any other comments, please write them here.</p>'+
        '<p style="text-align:left;"><textarea name="comments" rows="3" style="width:90%;"></textarea></p>'+
        '',
    data: {trial_name: 'exit_survey'},
};
timeline.push(exit_survey);

jsPsych.run(timeline);