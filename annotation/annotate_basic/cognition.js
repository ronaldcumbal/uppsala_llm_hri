const success_url = 'https://app.prolific.com/submissions/complete?cc=C6V6W902'

var jsPsych = initJsPsych({
    show_progress_bar: true,
    override_safe_mode: true,
    on_finish: function () {
        window.location.href = success_url;
    }
})

// ----->>>>> Paste code below this line from test HTML

var required_flag = true
    var timeline = [];

    var preload = {
        type: jsPsychPreload,
        images: [
            'example0.png',
            'example1.png',
            'example2.png',
            'example3.png',
            'example4.png'
        ]
    };
    timeline.push(preload);

    /* ============================= Prolific ============================= */
    var prolific = {
      type: jsPsychSurveyText,
      preamble: '<h2>Welcome to the experiment!</h2><p>If you come from <strong>Prolific</strong>, please write your unique Prolific ID here.<br> Otherwise, continue to the next page.</p>',
      questions: [
        {prompt: 'Prolific Id:', 
        required: false,
        name: 'prolific_id'}
      ],
      data: {trial_name: 'prolific_id'},
    }
    timeline.push(prolific);

    /* ============================= Instructions ============================= */
    
    var intructions = {
    type: jsPsychInstructions,
    pages: [
      '<style>p {text-align: justify}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 850px;}</style>'+
      '<div class="textbox">'+
      '<h3>Objective </h3>'+
      '<p>'+
      'The purpose of this study is to understand how people interpret human interactions. ' +
      'You will watch a series of short videos, each showing <strong>someone interacting</strong> with the person who recorded the video. ' +
      '<br>' +
      '<br>' +
      'Your task is to do two things for each video: ' +
      '<br>' +
      "1. Describe what you think the <strong>person interacting</strong> is trying to do or communicate (the <strong>intention</strong>)." +
      '<br>' +
      '2. Provide a brief description of the <strong>context</strong> in which the interaction takes place.' +
      '</p>'+
      '</div>',

      '<style>p {text-align: justify}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 840px;}</style>'+
      '<div class="textbox">'+
      '<h3>Requirements </h3>'+
      '<p>'+
      'In your <strong>descriptions</strong>, please use the following labels: ' +
      '<br>' +
      '<br>' +
      '<strong>#C</strong>: when describing the <strong>camera wearer</strong> (person recording the video with a head-mounted camera)' +
      '<br>' +
      '<strong>#O</strong>: when describing actions performed by the <strong>other person interacting</strong> in the scene. ' +
      '<br>' +
      '<br>' +
      'For example:' +
      '<br>' +
      '<img src="example0.png"></img>'+
      '</p>'+
      '</div>',

      '<style>p {text-align: center}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 840px;}</style>'+
      '<div class="textbox">'+
      '<p>'+
      'Before you begin, we’ll review a few examples to get you familiar with the task and interface: ' +
      '</p>'+
      '</div>',

      '<style>p {text-align: justify}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 1024px;}</style>'+
      '<div class="textbox">'+
      '<h3>Example 1: </h3>'+
      '<img src="example1.png"></img>'+
      '<p>'+
      ''+
      ''+
      '</p>'+
      '<p>'+
      'In this example, a single person is clearly visible on screen. '+
      'You can see that this person (#O) is handing a card to the camera wearer (#C). ' +
      'The context shows that they are interacting inside a living room. ' +
      '</p>'+
      '</div>',

      '<style>p {text-align: justify}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 1024px;}</style>'+
      '<div class="textbox">'+
      '<h3>Example 2: </h3>'+
      '<img src="example2.png"></img>'+
      '<p>'+
      ''+
      ''+
      '</p>'+
      '<p>'+
      'This one is a bit more complex, as multiple people are present. '+
      'In these cases, focus on the person looking directly at the camera. '+
      'Typically this is the person interacting with #C, and they’re usually the main focus around the middle of the video. '+
      '</p>'+
      '</div>',

      '<style>p {text-align: justify}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 1024px;}</style>'+
      '<div class="textbox">'+
      '<h3>Example 3: </h3>'+
      '<img src="example3.png"></img>'+
      '<p>'+
      ''+
      ''+
      '</p>'+
      '<p>'+
      'This video is a bit tricky. '+
      'The person is barely visible, but you can still hear the interaction. '+
      'Even with limited visuals, you should try to infer the person’s intention and the overall context from the audio. '+
      ''+
      '</p>'+
      '</div>',


      '<style>p {text-align: justify}</style>'+
      '<style>img {max-width: 100%; max-height: 100%;}</style>'+
      '<style>.textbox {width: 1024px;}</style>'+
      '<div class="textbox">'+
      '<h3>Example 4: </h3>'+
      '<img src="example4.png"></img>'+
      '<p>'+
      ''+
      ''+
      '</p>'+
      '<p>'+
      'The quality of the video is low, making it harder to interpret. '+
      'As with the previous example, do your best to describe the interaction. '+
      "If you're unsure about the intention or context, please skip the video by selecting the <strong>Skip</strong> option. "+
      'You’ll then be asked to briefly explain why the video was unclear to you. ' +
      '</p>'+
      '</div>',

      "Let’s start with the annotation."
      ],
        show_clickable_nav: true,
        allow_backward: false,
        button_label_next: 'Continue',
    }
    timeline.push(intructions);

    /* ============================= Task ============================= */

    var validation_videos = [
      "edea7771-89a8-4bcf-99c2-f102aa3d51dd_36990.mp4",
      "94c56fdd-9245-42db-8c58-252074e1521c_56623.mp4",
      "c8b61175-cfa4-44f9-a88b-76609ffa4ca3_57061.mp4",
    ];

    var data_videos = [
      "07774a7c-c5ed-4133-991f-926a316068a4_118.mp4",
      "5204624f-90ac-47bf-a71a-486d43f4351c_225.mp4",
      "50895f7f-5667-4b43-94c1-d7b4b9e238e7_111.mp4",
      "06592f50-7f29-4ca9-94a9-1b6c389d50ea_200.mp4",
      "dce1d8b4-2c29-4fec-a343-77885899c434_30.mp4",
      "ae6466f8-1a53-4cb3-8718-96f78f391b4b_0.mp4",
      "a6ea1a46-a7c4-41c8-b26d-08703ecd7690_192.mp4",
      "0d4efcc9-a336-46f9-a1db-0623b5be2869_196.mp4",
      "32464967-b0f2-4b2e-a57c-7f774e7056a7_295.mp4",
      "3aed7bbd-75f0-45ef-b9cc-7f5ff6f7fbbf_141.mp4",
      "1f12c871-9b3a-4611-a2b4-9c39059052a4_100.mp4",
      "e37c82c5-483f-4e68-8bf3-2a22dcef0b1e_110.mp4",
      "228eb504-c1b3-43b9-818e-2f7e539a259b_138.mp4",
      "6798a65f-d79a-413b-baca-115f09caf0c4_33.mp4",
      "57b6455e-2acb-4076-b8c4-1f2cdae97185_162.mp4",
      "de82a6f3-1a55-4996-ae68-35032678ff66_134.mp4",
      "b171dcaf-94f2-4bcd-8695-b2648bed60e1_14.mp4",
    ];

    // Insert validation videos at equally spaced locations
    var total_slots = data_videos.length + validation_videos.length;
    var interval = Math.floor(total_slots / validation_videos.length);
    var offset = Math.floor(interval / 2);

    for (var i = 0; i < validation_videos.length; i++) {
      var insertIndex = i * interval + offset;
      data_videos.splice(insertIndex, 0, validation_videos[i]);
    }

    for (var i = 0; i < data_videos.length; i++) {
      let videoFile = data_videos[i];

      let annotation_trial = {
        type: jsPsychSurvey,
        survey_json: {
          showQuestionNumbers: false,
          completeText: 'Continue',
          elements: [
            {
              type: 'html',
              name: 'video_' + videoFile,
              title: 'Video Annotation',
              description: 'Please watch the entire video before answering the questions below.',
              html: "<video width='768px' height='512px' controls><source src='" + videoFile + "' type='video/mp4'></video>",
              startWithNewLine: false,
            },
            {
              type: "radiogroup",
              name: "skip_video",
              title: "Do you want to skip this video because you're unsure?",
              isRequired: required_flag,
              choices: ["No", "Yes"],
              defaultValue: "No",
              colCount: 4,
              data: { trial_name: 'skip_video' },
            },
            {
              type: "comment",
              name: "ans_intention",
              title: "What do you think is the intention of the main person interacting in the video?",
              description: "Use #C: for the camera wearer and #O: for the other interacting person in the scene.",
              rows: 1,
              autoGrow: true,
              allowResize: false, 
              visibleIf: "{skip_video} = 'No'",
              data: { trial_name: 'ans_intention' }
            },
            {
              type: "comment",
              name: "ans_context",
              title: "What is the context or situation shown in the video?",
              description: "Use #C: for the camera wearer and #O: for the other interacting person in the scene.",
              rows: 1,
              autoGrow: true,
              allowResize: false, 
              visibleIf: "{skip_video} = 'No'",
              data: { trial_name: 'ans_context' }
            },
            {
              type: "comment",
              name: "skip_reason",
              title: "Why did you skip this video?",
              description: "Please briefly explain why you were unsure about this video.",
              visibleIf: "{skip_video} = 'Yes'",
              rows: 1,
              autoGrow: true,
              allowResize: false,
              data: { trial_name: 'skip_reason' }
            }
          ]
        },
        on_finish: function(data) {
          const skipped = data.response.skip_video === "Yes";
          console.log("Skipped video? ", skipped);
          if (skipped) {
            console.log("Reason: ", data.response.skip_reason);
          }
        }
      };

      timeline.push(annotation_trial);
    }


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
          '<p>If you have any comments, please write them here.</p>'+
          '<p style="text-align:left;"><textarea name="comments" rows="3" style="width:100%;"></textarea></p>'+
          '',
      data: {trial_name: 'exit_survey'},
    };
    timeline.push(exit_survey);

    jsPsych.run(timeline);