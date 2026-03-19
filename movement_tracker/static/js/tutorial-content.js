/**
 * Tutorial content data for the Movement Tracker tutorial series.
 *
 * Each tutorial is an object with:
 *   id        – integer (1-based, matches ?id= param)
 *   title     – short display title
 *   subtitle  – one-line description
 *   time      – estimated reading/doing time
 *   series    – "beginner" or "advanced"
 *   steps     – array of { title, body (HTML string), tips[] }
 */
window.TUTORIALS = [
    // ── 1. Viewing a Video ──────────────────────────────────
    {
        id: 1,
        title: 'Viewing a Video',
        subtitle: 'Browse and play back videos with zoom, speed, and camera controls.',
        time: '~3 min',
        series: 'beginner',
        steps: [
            {
                title: 'Open the Videos page',
                body: 'Click <a href="/videos"><strong>Videos</strong></a> in the navigation bar at the top of any page.',
                tips: [],
            },
            {
                title: 'Select a subject',
                body: 'Use the <strong>subject dropdown</strong> in the top toolbar. If the sample video has been added, select the subject that contains it (e.g.&nbsp;"Con01"). The first subject with videos is selected automatically.',
                tips: [],
            },
            {
                title: 'Choose a trial',
                body: 'Trial buttons appear in the toolbar after selecting a subject. Click a button (e.g.&nbsp;<strong>R1</strong>) to load that trial\u2019s video.',
                tips: [],
            },
            {
                title: 'Play and navigate frames',
                body: 'Press <kbd>Space</kbd> to play/pause. Step frame-by-frame with <kbd>\u2190</kbd> / <kbd>\u2192</kbd> (or <kbd>A</kbd> / <kbd>S</kbd>). The timeline slider at the bottom lets you scrub to any frame.',
                tips: [],
            },
            {
                title: 'Adjust playback speed',
                body: 'Use the <strong>speed slider</strong> in the bottom toolbar. Speeds range from 0.1\u00d7 to 120\u00d7. The current speed is displayed next to the slider.',
                tips: [],
            },
            {
                title: 'Zoom and pan',
                body: '<strong>Scroll wheel</strong> zooms in/out at the cursor position. <strong>Click and drag</strong> to pan. Press <kbd>Z</kbd> to reset the view.',
                tips: [],
            },
            {
                title: 'Toggle cameras',
                body: 'For stereo videos, press <kbd>E</kbd> to switch between OS (left) and OD (right) camera views. The current camera is shown on the toggle button.',
                tips: [
                    'The <strong>Browse\u2026</strong> button lets you preview any MP4 file from disk without adding it as a subject.',
                ],
            },
        ],
    },

    // ── 2. Adding a Subject ─────────────────────────────────
    {
        id: 2,
        title: 'Adding a Subject',
        subtitle: 'Use the onboarding wizard to create a subject from raw video.',
        time: '~5 min',
        series: 'beginner',
        steps: [
            {
                title: 'Start onboarding',
                body: 'On the <a href="/"><strong>Dashboard</strong></a>, click <strong>+ Add Subject</strong>. You can also navigate directly to the <a href="/onboarding">Onboarding</a> page.',
                tips: [],
            },
            {
                title: 'Enter a subject name',
                body: 'Type an alphanumeric identifier such as <code>Con01</code> or <code>MSA20</code>. No spaces \u2014 underscores are allowed. Press <strong>Confirm</strong> to continue.',
                tips: [],
            },
            {
                title: 'Browse to the source video',
                body: 'The file browser shows your configured video directory. Navigate folders and click a video file (e.g.&nbsp;<code>Con01_R1.mp4</code>) to select it. Metadata like resolution, FPS, and duration appear automatically.',
                tips: ['Double-click a file to select it immediately.'],
            },
            {
                title: 'Set trim points',
                body: 'Use the video player to find the start of the movement. Press <kbd>I</kbd> to mark the <strong>In point</strong>. Seek to the end of the movement and press <kbd>O</kbd> to mark the <strong>Out point</strong>.',
                tips: [],
            },
            {
                title: 'Label the trial and add segment',
                body: 'Enter a trial label (e.g.&nbsp;<code>R1</code>, <code>L2</code>, or any name you like) in the trial name field. Suggestions appear but you can type freely. Click <strong>Add Segment</strong>.',
                tips: ['Repeat steps 3\u20135 for each additional trial in the same session, even from different source videos.'],
            },
            {
                title: 'Review and process',
                body: 'The segment list shows all trials with their time ranges and output filenames. Optionally check <strong>"Blur faces (de-identify)"</strong>. Click <strong>Trim &amp; Create Subject</strong> to start processing.',
                tips: [
                    'Processing runs in the background \u2014 a progress bar shows the current step.',
                    'When finished, you\u2019re redirected to the Dashboard where the new subject appears.',
                ],
            },
        ],
    },

    // ── 3. De-identifying (Blurring Faces) ──────────────────
    {
        id: 3,
        title: 'De-identifying (Blurring Faces)',
        subtitle: 'Blur faces in videos for privacy before sharing or analysis.',
        time: '~3 min',
        series: 'beginner',
        steps: [
            {
                title: 'Open the Processing page',
                body: 'Click <a href="/remote"><strong>Processing</strong></a> in the nav bar.',
                tips: [],
            },
            {
                title: 'Select the Blur step',
                body: 'In the <strong>Step</strong> dropdown at the top of the Launch Job section, choose <strong>Blur</strong>.',
                tips: [],
            },
            {
                title: 'Select subjects',
                body: 'Check the boxes next to the subject(s) you want to de-identify. Use <strong>Select All</strong> for batch processing.',
                tips: [],
            },
            {
                title: 'Choose execution target and submit',
                body: 'Select <strong>Local CPU</strong> (fine for a few videos) or <strong>Local GPU</strong> if available. Click <strong>Submit</strong>.',
                tips: [],
            },
            {
                title: 'Monitor progress',
                body: 'The job appears in the <strong>Queue</strong> section. When the status changes to complete (\u2713), blurring is done. Original videos are preserved; blurred copies go to a <code>deidentified/</code> subdirectory.',
                tips: [],
            },
            {
                title: 'Prefer de-identified videos',
                body: 'Go to <a href="/settings"><strong>Settings</strong></a> and check <strong>"Show deidentified videos"</strong>. The video viewer and labeler will now show blurred versions by default wherever available.',
                tips: [
                    'You can also blur during onboarding by checking "Blur faces" in Step 4 \u2014 this runs the same process as part of subject creation.',
                ],
            },
        ],
    },

    // ── 4. Fitting MediaPipe Labels ─────────────────────────
    {
        id: 4,
        title: 'Fitting MediaPipe Labels',
        subtitle: 'Run automatic hand landmark detection to generate starting labels.',
        time: '~3 min',
        series: 'beginner',
        steps: [
            {
                title: 'Open the Processing page',
                body: 'Click <a href="/remote"><strong>Processing</strong></a> in the nav bar.',
                tips: [],
            },
            {
                title: 'Select the MediaPipe step',
                body: 'Choose <strong>MediaPipe</strong> from the Step dropdown.',
                tips: [],
            },
            {
                title: 'Select subjects and submit',
                body: 'Check your subject(s) and click <strong>Submit</strong>. MediaPipe detects 21 hand landmarks per frame for each camera view.',
                tips: [],
            },
            {
                title: 'Wait for completion',
                body: 'The job runs in the background. Processing time depends on the number of frames \u2014 typically a few minutes per trial on CPU.',
                tips: [],
            },
            {
                title: 'View MediaPipe ghosts in the labeler',
                body: 'Open the <a href="/labeling-select"><strong>DLC</strong></a> page and select your subject. You should see <strong>cyan ghost markers</strong> overlaid on the video \u2014 these are MediaPipe\u2019s predictions.',
                tips: [
                    'MediaPipe ghosts are a starting point \u2014 you\u2019ll correct them manually in the next tutorial.',
                    'The video viewer also uses MediaPipe hints to auto-select the best camera and zoom to the hand region.',
                ],
            },
        ],
    },

    // ── 5. Labeling with DLC & Training ─────────────────────
    {
        id: 5,
        title: 'Labeling with DLC & Training',
        subtitle: 'Manually annotate keypoints, then train a DeepLabCut model.',
        time: '~10 min',
        series: 'beginner',
        steps: [
            {
                title: 'Open the DLC labeler',
                body: 'Click <a href="/labeling-select"><strong>DLC</strong></a> in the nav bar. Select your subject from the dropdown. If prompted, choose <strong>Initial</strong> mode to start a fresh labeling session.',
                tips: [],
            },
            {
                title: 'Understand the interface',
                body: 'The main area shows the video frame. The sidebar lists <strong>bodyparts</strong> (e.g.&nbsp;thumb, index) in order. Cyan dots are MediaPipe ghosts. The timeline at the bottom shows which frames have labels (green dots).',
                tips: [],
            },
            {
                title: 'Place keypoints',
                body: 'Click on the video frame to place a label for the current bodypart. The active bodypart is highlighted in the sidebar. After placing a label, the next bodypart activates automatically.',
                tips: [
                    'Click near a cyan MediaPipe ghost to accept its position, or click elsewhere to correct it.',
                ],
            },
            {
                title: 'Edit labels',
                body: '<strong>Drag</strong> an existing label to reposition it. <strong>Right-click</strong> a label to delete it. Use <kbd>Ctrl+Z</kbd> to undo and <kbd>Ctrl+Shift+Z</kbd> to redo.',
                tips: [],
            },
            {
                title: 'Label multiple frames',
                body: 'Navigate to different frames using <kbd>\u2190</kbd> / <kbd>\u2192</kbd> and label each one. Aim for <strong>15\u201320 frames</strong> spread across the trial, covering different hand positions (open, closed, mid-movement).',
                tips: [
                    'Press <kbd>D</kbd> / <kbd>F</kbd> to jump to the next/previous already-labeled frame.',
                    'Spreading labels across diverse poses gives the model a better training set.',
                ],
            },
            {
                title: 'Save and commit',
                body: 'Click <strong>Save &amp; Commit</strong> in the sidebar. This writes your labels to the DLC training data directory. The subject\u2019s stage updates to "labeled" on the Dashboard.',
                tips: [],
            },
            {
                title: 'Train a DLC model',
                body: 'Go to <a href="/remote"><strong>Processing</strong></a>, select <strong>Train</strong>, check your subject, and submit. Training creates a neural network from your manual labels.',
                tips: [
                    'Training takes 30\u201360 minutes on a GPU, longer on CPU. A warning banner appears if you select CPU.',
                    'If you have a remote GPU server configured in Settings, select <strong>Remote</strong> as the execution target.',
                ],
            },
        ],
    },

    // ── 6. DLC Analysis & Viewing Predictions ───────────────
    {
        id: 6,
        title: 'DLC Analysis & Viewing Predictions',
        subtitle: 'Run your trained model on all frames and view the predictions.',
        time: '~5 min',
        series: 'beginner',
        steps: [
            {
                title: 'Run DLC analysis',
                body: 'After training completes, go to <a href="/remote"><strong>Processing</strong></a>. Select <strong>Analyze v1</strong>, check your subject, and submit. This runs the trained DLC model on all video frames.',
                tips: [],
            },
            {
                title: 'View DLC predictions in the labeler',
                body: 'Open the <a href="/labeling-select"><strong>DLC</strong></a> page. Alongside your manual labels, you\u2019ll see DLC prediction markers on every frame \u2014 not just the ones you labeled.',
                tips: ['The model\u2019s confidence is reflected in the marker size and opacity.'],
            },
            {
                title: 'Spot-check across the trial',
                body: 'Scrub through the video and look for frames where the predicted keypoints drift off the hand or land on the wrong finger. Note these \u2014 you\u2019ll fix them in the next tutorial (Refinement).',
                tips: [
                    'Pay attention to frames where the hand is partially occluded or rapidly moving \u2014 models often struggle there.',
                ],
            },
            {
                title: 'Check the Dashboard',
                body: 'Return to the <a href="/"><strong>Dashboard</strong></a>. Your subject\u2019s stage badge should now say <strong>"analyzed"</strong>, confirming that DLC analysis is complete.',
                tips: [],
            },
        ],
    },

    // ── 7. Refinement & Correction ──────────────────────────
    {
        id: 7,
        title: 'Refinement & Correction',
        subtitle: 'Fix DLC prediction errors and retrain for better accuracy.',
        time: '~8 min',
        series: 'advanced',
        steps: [
            {
                title: 'Open a refinement session',
                body: 'Go to <a href="/labeling-select"><strong>DLC</strong></a> and select your subject. Choose <strong>Refinement</strong> mode. This loads DLC\u2019s predictions as editable labels so you can correct mistakes.',
                tips: [],
            },
            {
                title: 'Find frames that need correction',
                body: 'Scrub through the trial using <kbd>\u2190</kbd> / <kbd>\u2192</kbd>. Look for keypoints that are visibly misplaced \u2014 on the wrong finger, off the hand entirely, or jittering between frames.',
                tips: [
                    'The <strong>confidence heatmap</strong> in the timeline highlights low-confidence frames in red \u2014 start there.',
                    'Press <kbd>D</kbd> / <kbd>F</kbd> to jump between frames that already have corrections.',
                ],
            },
            {
                title: 'Correct misplaced keypoints',
                body: 'Click a bodypart in the sidebar to select it, then click the correct position on the video to move the label. You can also <strong>drag</strong> existing labels directly. <strong>Right-click</strong> a label to delete it if the bodypart is not visible.',
                tips: [],
            },
            {
                title: 'Focus on outlier frames',
                body: 'You don\u2019t need to fix every frame \u2014 focus on the <strong>worst outliers</strong>. Correcting 10\u201320 frames where the model is clearly wrong gives the biggest improvement per frame labeled.',
                tips: [
                    'Frames where the hand changes direction (opening to closing) are often the hardest for the model.',
                ],
            },
            {
                title: 'Save corrections and commit',
                body: 'Click <strong>Save &amp; Commit</strong>. Your corrections are merged into the training dataset alongside your original labels.',
                tips: [],
            },
            {
                title: 'Retrain with corrections',
                body: 'Go to <a href="/remote"><strong>Processing</strong></a>, select <strong>Train</strong>, and submit. The model will now train on both your original labels and your corrections, producing better predictions.',
                tips: [
                    'After retraining, run <strong>Analyze v1</strong> again to see the improved predictions.',
                    'You can repeat the refine\u2013retrain cycle multiple times. Each round typically improves accuracy, especially for difficult frames.',
                ],
            },
            {
                title: 'Re-analyze and verify',
                body: 'Run <strong>Analyze v1</strong> again on your subject. Open the labeler and compare the new predictions to the old ones \u2014 the corrected frames should now track much more accurately.',
                tips: [],
            },
        ],
    },

    // ── 8. Event Detection & Correction ─────────────────────
    {
        id: 8,
        title: 'Event Detection & Correction',
        subtitle: 'Automatically detect movement events and manually correct them.',
        time: '~7 min',
        series: 'advanced',
        steps: [
            {
                title: 'Run event detection',
                body: 'Go to <a href="/remote"><strong>Processing</strong></a>. Select <strong>Events</strong> from the Step dropdown, check your subject, and submit. The algorithm detects <strong>open</strong>, <strong>peak</strong>, and <strong>close</strong> events from the distance traces.',
                tips: [],
            },
            {
                title: 'View detected events in Results',
                body: 'Open <a href="/results"><strong>Results</strong></a> and select your subject. The <strong>Distances</strong> tab now shows colored vertical markers for each detected event overlaid on the distance trace.',
                tips: [
                    'Open events are green, peak events are yellow, and close events are red \u2014 matching the colors configured in Settings.',
                ],
            },
            {
                title: 'Understand the event types',
                body: '<strong>Open</strong> marks the start of a finger-opening movement. <strong>Peak</strong> marks the maximum opening amplitude. <strong>Close</strong> marks the return to a closed position. Together, they define one complete movement cycle.',
                tips: [],
            },
            {
                title: 'Correct event positions',
                body: 'In the Results page, click an event marker to select it. <strong>Drag</strong> it left or right along the timeline to adjust its frame position. The event snaps to the nearest frame.',
                tips: [
                    'If an event was detected in the wrong place, drag it to the correct frame. If an event is missing, right-click the timeline to add one.',
                ],
            },
            {
                title: 'Add or remove events',
                body: 'To add a missing event, <strong>right-click</strong> on the distance trace at the desired frame and select the event type from the context menu. To remove a false detection, select the event marker and press <kbd>Delete</kbd>.',
                tips: [],
            },
            {
                title: 'Save corrected events',
                body: 'Click <strong>Save Events</strong> to persist your corrections. The movement parameters in the <strong>Movements</strong> tab will update to reflect the corrected event boundaries.',
                tips: [
                    'Accurate event boundaries are critical for reliable amplitude, velocity, and inter-movement interval calculations.',
                ],
            },
        ],
    },

    // ── 9. Viewing Results ──────────────────────────────────
    {
        id: 9,
        title: 'Viewing Results',
        subtitle: 'Explore distance traces, movement parameters, and export data.',
        time: '~5 min',
        series: 'advanced',
        steps: [
            {
                title: 'Open the Results page',
                body: 'Click <a href="/results"><strong>Results</strong></a> in the nav bar and select your subject from the dropdown.',
                tips: [],
            },
            {
                title: 'Explore the Distances tab',
                body: 'The <strong>Distances</strong> tab shows line plots of 3D distances (e.g.&nbsp;thumb-to-index opening) over time for each trial. Each trial gets its own plot. Event markers (if detected) appear as colored vertical lines.',
                tips: ['Check <strong>Lock Y-axis</strong> to use the same scale across all trial plots for easier comparison.'],
            },
            {
                title: 'Explore the Movements tab',
                body: 'The <strong>Movements</strong> tab computes per-event parameters: <strong>amplitude</strong> (max opening), <strong>velocity</strong> (peak opening/closing speed), <strong>inter-movement interval</strong>, and more. Use the checkboxes to toggle which parameters are visible.',
                tips: [],
            },
            {
                title: 'Compare across trials',
                body: 'Use the trial selector to switch between trials, or view all trials side by side. Look for trends like decreasing amplitude or increasing interval \u2014 these may indicate fatigue or disease progression.',
                tips: [],
            },
            {
                title: 'Export data',
                body: 'Click <strong>Export</strong> to download the results as a CSV file. This includes per-event parameters for each trial, ready for statistical analysis in R, Python, or Excel.',
                tips: [
                    'The exported CSV uses one row per movement event, with columns for amplitude, velocity, duration, and interval.',
                ],
            },
            {
                title: 'Check the Dashboard',
                body: 'Return to the <a href="/"><strong>Dashboard</strong></a>. Your subject\u2019s stage badge should now reflect the full pipeline completion. From here, you can proceed to MANO fitting for 3D hand model reconstruction.',
                tips: [],
            },
        ],
    },

    // ── 10. MANO Fitting & Viewing ──────────────────────────
    {
        id: 10,
        title: 'MANO Fitting & Viewing',
        subtitle: 'Fit a 3D hand model to your tracked keypoints and visualize the results.',
        time: '~5 min',
        series: 'advanced',
        steps: [
            {
                title: 'Prerequisites',
                body: 'MANO fitting requires completed DLC analysis (keypoints on all frames) and stereo calibration. Ensure your subject has been analyzed and a calibration file is assigned in Settings.',
                tips: [],
            },
            {
                title: 'Run MANO fitting',
                body: 'Go to <a href="/remote"><strong>Processing</strong></a>. Select <strong>MANO Fit</strong> from the Step dropdown, check your subject, and submit. This is a GPU-accelerated job that fits the MANO 3D hand model to your 2D keypoint detections across all frames.',
                tips: [
                    'MANO fitting is computationally intensive \u2014 a GPU is strongly recommended. On CPU it may take significantly longer.',
                    'If you have a remote GPU configured, select <strong>Remote</strong> as the execution target.',
                ],
            },
            {
                title: 'Monitor the fitting job',
                body: 'The job appears in the <strong>Queue</strong>. MANO fitting typically takes a few minutes per trial on GPU. The progress indicator shows which trial is currently being processed.',
                tips: [],
            },
            {
                title: 'Open the MANO viewer',
                body: 'Once fitting completes, click <a href="/mano"><strong>MANO</strong></a> in the nav bar and select your subject. The viewer shows a 3D hand mesh overlaid on the video frame.',
                tips: [],
            },
            {
                title: 'Navigate the 3D view',
                body: 'The MANO viewer has the same frame navigation as the video viewer (<kbd>\u2190</kbd> / <kbd>\u2192</kbd>, <kbd>Space</kbd> to play). Additionally, you can <strong>rotate</strong> the 3D view by dragging, and <strong>zoom</strong> with the scroll wheel.',
                tips: [
                    'Toggle between the video overlay and a standalone 3D mesh view using the view mode selector.',
                ],
            },
            {
                title: 'Assess fit quality',
                body: 'Scrub through the trial and check that the 3D mesh follows the actual hand movement. The mesh should align with the fingers during opening and closing. Misalignment usually indicates DLC prediction errors upstream.',
                tips: [
                    'If the mesh doesn\u2019t align well, go back to the Refinement tutorial to correct DLC predictions, retrain, re-analyze, and re-fit MANO.',
                ],
            },
        ],
    },
];
