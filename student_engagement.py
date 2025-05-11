# student_engagement_system.py
import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime, date, timedelta
import os
import sqlite3
from gaze_tracking.gaze_tracking import GazeTracking
import plotly.express as px
import plotly.graph_objects as go
import uuid


# Initialize database
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 timestamp TEXT NOT NULL,
                 date TEXT NOT NULL,
                 UNIQUE(name, date) ON CONFLICT IGNORE)"""
    )  # Ensures one record per student per day

    # Create assignment table
    c.execute(
        """CREATE TABLE IF NOT EXISTS assignments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 student_name TEXT NOT NULL,
                 assignment_name TEXT NOT NULL,
                 file_path TEXT NOT NULL,
                 submission_date TEXT NOT NULL,
                 feedback TEXT,
                 grade REAL)"""
    )

    # Create engagement table
    c.execute(
        """CREATE TABLE IF NOT EXISTS engagement
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 timestamp TEXT NOT NULL,
                 date TEXT NOT NULL,
                 status TEXT NOT NULL)"""
    )

    conn.commit()
    conn.close()


init_db()


# Database functions
def save_to_db(name):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    today = date.today().isoformat()

    # Check if already marked today
    c.execute(
        "SELECT COUNT(*) FROM attendance WHERE name = ? AND date = ?", (name, today)
    )
    already_marked = c.fetchone()[0] > 0

    if not already_marked:
        # Insert new attendance record
        c.execute(
            "INSERT INTO attendance (name, timestamp, date) VALUES (?,?,?)",
            (name, timestamp, today),
        )
        conn.commit()
        conn.close()
        return True
    else:
        conn.close()
        return False


def get_attendance_from_db():
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql_query(
        """
        SELECT name, timestamp, date 
        FROM attendance 
        ORDER BY date DESC, timestamp DESC
    """,
        conn,
    )
    conn.close()
    return df


def save_engagement_to_db(name, status):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    today = date.today().isoformat()

    c.execute(
        "INSERT INTO engagement (name, timestamp, date, status) VALUES (?,?,?,?)",
        (name, timestamp, today, status),
    )
    conn.commit()
    conn.close()


def get_engagement_from_db():
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql_query(
        """
        SELECT name, timestamp, date, status
        FROM engagement 
        ORDER BY date DESC, timestamp DESC
    """,
        conn,
    )
    conn.close()
    return df


def save_assignment_to_db(student_name, assignment_name, file_path):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    now = datetime.now()
    submission_date = now.strftime("%Y-%m-%d %H:%M:%S")

    c.execute(
        "INSERT INTO assignments (student_name, assignment_name, file_path, submission_date, feedback, grade) VALUES (?,?,?,?,?,?)",
        (student_name, assignment_name, file_path, submission_date, "", None),
    )
    conn.commit()
    conn.close()


def get_assignments_from_db():
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql_query(
        """
        SELECT id, student_name, assignment_name, submission_date, feedback, grade 
        FROM assignments 
        ORDER BY submission_date DESC
    """,
        conn,
    )
    conn.close()
    return df


def update_assignment_feedback(assignment_id, feedback, grade):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute(
        "UPDATE assignments SET feedback = ?, grade = ? WHERE id = ?",
        (feedback, grade, assignment_id),
    )
    conn.commit()
    conn.close()


# Initialize gaze tracking and load known faces
gaze = GazeTracking()
known_face_encodings = []
known_face_names = []

# Load student images
FACE_DIR = "known_faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

for filename in os.listdir(FACE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(FACE_DIR, filename)
        try:
            image = face_recognition.load_image_file(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(image)

            if face_locations:
                encodings = face_recognition.face_encodings(image, face_locations)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(os.path.splitext(filename)[0])
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")


def log_attendance(name):
    """Mark attendance and return whether this is a new attendance mark"""
    recorded = save_to_db(name)
    if recorded:
        now = datetime.now()
        attendance_id = f"{name}_{now.date().isoformat()}"
        if attendance_id not in st.session_state.already_notified:
            st.session_state.last_attendance = {
                "name": name,
                "date": now.date().isoformat(),
            }
            st.session_state.already_notified.add(attendance_id)
            if "attendance_last_shown_date" not in st.session_state:
                st.session_state.attendance_last_shown_date = None
            return True
    return False


# Streamlit UI
st.set_page_config(
    page_title="AI-Powered Student Engagement Platform",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI-Powered Student Engagement & Monitoring Platform")
today = date.today().isoformat()
if (
    "current_notification_date" not in st.session_state
    or st.session_state.current_notification_date != today
):
    st.session_state.already_notified = set()
    st.session_state.current_notification_date = today

# Initialize session state for student management
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "last_attendance" not in st.session_state:
    st.session_state.last_attendance = None
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "already_notified" not in st.session_state:
    st.session_state.already_notified = set()

page = st.sidebar.selectbox(
    "Choose Module",
    [
        "Live Engagement Monitor",
        "Attendance Log",
        "Performance Dashboard",
        "Submit Assignment",
        "Accessibility Tools",
        "Student Management",
    ],
)

if page == "Live Engagement Monitor":
    st.header("🎥 Live Facial Recognition & Gaze Tracking")

    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence_threshold = st.slider("Recognition Confidence", 0.1, 1.0, 0.6, 0.05)
    with col2:
        engagement_tracking = st.checkbox("Track Engagement", value=True)
    with col3:
        engagement_interval = st.number_input(
            "Save Engagement Data (sec)", min_value=30, value=60, step=10
        )

    FRAME_WINDOW = st.image([])
    stop_button = st.button("Stop Camera")

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    # Last engagement timestamp
    last_engagement_time = datetime.now()

    while True and not stop_button:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if not face_locations:
            FRAME_WINDOW.image(frame, channels="BGR")
            continue

        try:
            # Process each face individually
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )

                if len(face_distances) > 0:  # Check if there are any face distances
                    best_match_index = np.argmin(face_distances)
                    if (
                        matches[best_match_index]
                        and face_distances[best_match_index] < confidence_threshold
                    ):
                        name = known_face_names[best_match_index]

                        # Check if already marked today
                        conn = sqlite3.connect("attendance.db")
                        c = conn.cursor()
                        today = date.today().isoformat()
                        c.execute(
                            "SELECT COUNT(*) FROM attendance WHERE name = ? AND date = ?",
                            (name, today),
                        )
                        already_marked = c.fetchone()[0] > 0
                        conn.close()

                # Scale back up face locations
                top, right, bottom, left = [coord * 4 for coord in face_location]

                # Create a unique ID for this attendance event
                attendance_id = f"{name}_{today}"

                # Change box color based on attendance status
                if name == "Unknown":
                    color = (0, 0, 255)  # Red for unknown
                    status_text = "UNKNOWN"
                elif already_marked:
                    color = (255, 255, 0)  # Yellow for already marked
                    status_text = "MARKED TODAY"
                else:
                    color = (0, 255, 0)  # Green for recognized
                    status_text = "READY TO MARK"

                    # Mark attendance if recognized and not already marked today
                    if name != "Unknown" and not already_marked:
                        if log_attendance(name):
                            # Update the status after marking
                            status_text = "MARKED NOW"
                            color = (0, 255, 0)  # Green

                            # # Add notification only if we haven't shown it yet
                            # if attendance_id not in st.session_state.already_notified:
                            #     st.toast(
                            #         f"✅ Attendance marked for {name} on {date.today().isoformat()}"
                            #     )
                            #     st.session_state.already_notified.add(attendance_id)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                )

                # Attendance status text
                cv2.putText(
                    frame,
                    f"ATTENDANCE: {status_text}",
                    (left + 6, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                )

            # Gaze tracking
            gaze.refresh(frame)
            gaze_text = "Not focused"
            engagement_status = "Disengaged"

            if gaze.is_blinking():
                gaze_text = "Blinking"
                engagement_status = "Neutral"
            elif gaze.is_right():
                gaze_text = "Looking right"
                engagement_status = "Distracted"
            elif gaze.is_left():
                gaze_text = "Looking left"
                engagement_status = "Distracted"
            elif gaze.is_center():
                gaze_text = "Looking center"
                engagement_status = "Engaged"

            cv2.putText(
                frame,
                f"Gaze: {gaze_text}",
                (20, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Status: {engagement_status}",
                (20, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (
                    (0, 255, 0)
                    if engagement_status == "Engaged"
                    else (
                        (255, 255, 0) if engagement_status == "Neutral" else (0, 0, 255)
                    )
                ),
                2,
            )

            # Save engagement data at intervals
            if (
                engagement_tracking
                and name != "Unknown"
                and (datetime.now() - last_engagement_time).total_seconds()
                > engagement_interval
            ):
                save_engagement_to_db(name, engagement_status)
                last_engagement_time = datetime.now()

            FRAME_WINDOW.image(frame, channels="BGR")

            # Show last attendance in sidebar
            # if st.session_state.last_attendance:
            #     st.sidebar.success(
            #         f"Last attendance marked:\n"
            #         f"Student: {st.session_state.last_attendance['name']}\n"
            #         f"Date: {st.session_state.last_attendance['date']}"
            #     )
            if (
                st.session_state.last_attendance
                and st.session_state.attendance_last_shown_date != today
            ):
                st.sidebar.success(
                    f"Last attendance marked:\n"
                    f"Student: {st.session_state.last_attendance['name']}\n"
                    f"Date: {st.session_state.last_attendance['date']}"
                )
                st.session_state.attendance_last_shown_date = today

        except Exception as e:
            st.error(f"Error processing faces: {str(e)}")
            continue

    # Release resources when stopping
    if video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()

elif page == "Attendance Log":
    st.header("📝 Daily Attendance Log")

    attendance_df = get_attendance_from_db()

    if not attendance_df.empty:
        # Convert to datetime for filtering
        attendance_df["date"] = pd.to_datetime(attendance_df["date"])

        # Date selector
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_date = st.date_input(
                "Select date to view",
                value=date.today(),
                min_value=pd.to_datetime(attendance_df["date"]).min().date(),
                max_value=date.today(),
            )

        with col2:
            view_mode = st.radio("View mode", ["Daily", "Weekly", "Monthly"])

        # Filter based on view mode
        if view_mode == "Daily":
            filter_start = selected_date
            filter_end = selected_date
            period_str = f"{selected_date.strftime('%A, %B %d, %Y')}"
        elif view_mode == "Weekly":
            # Find the Monday of the week containing selected_date
            filter_start = selected_date - timedelta(days=selected_date.weekday())
            filter_end = filter_start + timedelta(days=6)
            period_str = f"Week {filter_start.strftime('%Y-%m-%d')} to {filter_end.strftime('%Y-%m-%d')}"
        else:  # Monthly
            filter_start = selected_date.replace(day=1)
            if filter_start.month == 12:
                filter_end = filter_start.replace(
                    year=filter_start.year + 1, month=1, day=1
                ) - timedelta(days=1)
            else:
                filter_end = filter_start.replace(
                    month=filter_start.month + 1, day=1
                ) - timedelta(days=1)
            period_str = f"{filter_start.strftime('%B %Y')}"

        # Filter by selected period
        filtered_df = attendance_df[
            (attendance_df["date"].dt.date >= filter_start)
            & (attendance_df["date"].dt.date <= filter_end)
        ]

        # Display
        st.subheader(f"Attendance for {period_str}")

        if not filtered_df.empty:
            # Present/Absent summary
            all_students = set(known_face_names)
            present_students = set(filtered_df["name"])
            absent_students = all_students - present_students

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Present", len(present_students))
                st.dataframe(
                    pd.DataFrame(sorted(present_students), columns=["Present Students"])
                )

            with col2:
                st.metric("Absent", len(absent_students))
                st.dataframe(
                    pd.DataFrame(sorted(absent_students), columns=["Absent Students"])
                )

            # Detailed records
            st.subheader("Detailed Records")
            st.dataframe(filtered_df.sort_values(["date", "timestamp"]))

            # Export button
            st.download_button(
                "Export to CSV",
                filtered_df.to_csv(index=False),
                f"attendance_{filter_start}_to_{filter_end}.csv",
                "text/csv",
            )

            # Visual representation
            st.subheader("Attendance Visualization")

            # For daily view - just show who's present
            if view_mode == "Daily":
                fig = px.bar(
                    x=list(present_students),
                    y=[1] * len(present_students),
                    labels={"x": "Student", "y": "Present"},
                    title=f"Students Present on {selected_date}",
                )
                st.plotly_chart(fig)

            # For weekly/monthly - show attendance pattern
            else:
                # Create a pivot table with dates and student names
                pivot_df = pd.pivot_table(
                    filtered_df,
                    values="timestamp",
                    index=["date"],
                    columns=["name"],
                    aggfunc="count",
                ).fillna(0)

                # Convert to binary present/absent
                pivot_df = (pivot_df > 0).astype(int)

                # Create heatmap
                fig = px.imshow(
                    pivot_df.T,
                    labels=dict(x="Date", y="Student", color="Present"),
                    x=pivot_df.index.strftime("%Y-%m-%d"),
                    y=pivot_df.columns,
                    color_continuous_scale=["white", "green"],
                    title=f"Attendance Heatmap for {period_str}",
                )
                st.plotly_chart(fig)
        else:
            st.warning(f"No attendance records found for {period_str}")
    else:
        st.info("No attendance records found in database")

elif page == "Performance Dashboard":
    st.header("📈 Performance Dashboard")

    tab1, tab2, tab3 = st.tabs(
        ["Attendance Analysis", "Engagement Analysis", "Assignment Performance"]
    )

    with tab1:
        attendance_df = get_attendance_from_db()

        if not attendance_df.empty:
            # Process data
            df = attendance_df.copy()
            df["date"] = pd.to_datetime(df["date"])

            # Time period selection
            time_period = st.selectbox(
                "Select Time Period",
                ["Last Week", "Last Month", "Last 3 Months", "All Time"],
            )

            if time_period == "Last Week":
                start_date = date.today() - timedelta(days=7)
            elif time_period == "Last Month":
                start_date = date.today() - timedelta(days=30)
            elif time_period == "Last 3 Months":
                start_date = date.today() - timedelta(days=90)
            else:
                start_date = df["date"].min().date()

            filtered_df = df[df["date"].dt.date >= start_date]

            # Monthly attendance
            st.subheader("Attendance Trend")

            if time_period in ["Last Week", "Last Month"]:
                # For shorter periods, show daily trend
                daily = filtered_df.groupby([filtered_df["date"].dt.date]).size()
                fig = px.line(
                    x=daily.index,
                    y=daily.values,
                    markers=True,
                    labels={"x": "Date", "y": "Student Count"},
                    title="Daily Attendance",
                )
                st.plotly_chart(fig)
            else:
                # For longer periods, show monthly trend
                monthly = filtered_df.groupby(
                    [filtered_df["date"].dt.to_period("M")]
                ).size()
                fig = px.bar(
                    x=[str(period) for period in monthly.index],
                    y=monthly.values,
                    labels={"x": "Month", "y": "Total Attendance"},
                    title="Monthly Attendance Trend",
                )
                st.plotly_chart(fig)

            # Student attendance rates
            st.subheader("Student Attendance Rates")

            # Calculate date range
            if filtered_df.empty:
                st.warning("No data available for the selected period")
            else:
                # Calculate attendance rates
                total_days = (
                    filtered_df["date"].max().date() - filtered_df["date"].min().date()
                ).days + 1

                # Count school days (weekdays)
                school_days = sum(
                    1
                    for d in range(total_days)
                    if (filtered_df["date"].min().date() + timedelta(days=d)).weekday()
                    < 5
                )

                student_stats = (
                    filtered_df.groupby("name")
                    .agg(
                        days_present=("date", lambda x: len(set(d.date() for d in x))),
                    )
                    .assign(
                        attendance_rate=lambda x: (
                            x["days_present"] / max(1, school_days) * 100
                        ).round(1)
                    )
                    .sort_values("attendance_rate", ascending=False)
                )

                # Add missing students with 0% attendance
                missing_students = set(known_face_names) - set(student_stats.index)
                for student in missing_students:
                    student_stats.loc[student] = [0, 0.0]

                # Display as bar chart
                fig = px.bar(
                    student_stats.reset_index(),
                    x="name",
                    y="attendance_rate",
                    color="attendance_rate",
                    color_continuous_scale=["red", "yellow", "green"],
                    range_color=[0, 100],
                    labels={
                        "name": "Student",
                        "attendance_rate": "Attendance Rate (%)",
                    },
                    title="Student Attendance Rates",
                )
                st.plotly_chart(fig)

                # Display as table
                st.dataframe(
                    student_stats.sort_values("attendance_rate", ascending=False)
                )
        else:
            st.warning("No attendance data available for performance analysis")

    with tab2:
        engagement_df = get_engagement_from_db()

        if not engagement_df.empty:
            st.subheader("Student Engagement Analysis")

            engagement_df["date"] = pd.to_datetime(engagement_df["date"])
            engagement_df["timestamp"] = pd.to_datetime(engagement_df["timestamp"])

            # Select student
            selected_student = st.selectbox(
                "Select Student",
                ["All Students"] + sorted(engagement_df["name"].unique().tolist()),
            )

            if selected_student != "All Students":
                filtered_engagement = engagement_df[
                    engagement_df["name"] == selected_student
                ]
            else:
                filtered_engagement = engagement_df

            # Display engagement distribution
            engagement_counts = filtered_engagement["status"].value_counts()

            fig = px.pie(
                values=engagement_counts.values,
                names=engagement_counts.index,
                title=f"Engagement Distribution for {selected_student}",
                color=engagement_counts.index,
                color_discrete_map={
                    "Engaged": "green",
                    "Neutral": "yellow",
                    "Distracted": "orange",
                    "Disengaged": "red",
                },
            )
            st.plotly_chart(fig)

            # Show engagement over time for a specific student
            if selected_student != "All Students":
                st.subheader(f"Engagement Timeline for {selected_student}")

                # Create a time series
                timeline_df = filtered_engagement.sort_values("timestamp")

                # Convert status to numeric value for visualization
                status_map = {
                    "Engaged": 3,
                    "Neutral": 2,
                    "Distracted": 1,
                    "Disengaged": 0,
                }

                timeline_df["engagement_level"] = timeline_df["status"].map(status_map)

                fig = px.line(
                    timeline_df,
                    x="timestamp",
                    y="engagement_level",
                    markers=True,
                    labels={
                        "timestamp": "Time",
                        "engagement_level": "Engagement Level",
                    },
                    title=f"Engagement Timeline for {selected_student}",
                )

                # Add custom y-axis labels
                fig.update_layout(
                    yaxis=dict(
                        tickmode="array",
                        tickvals=[0, 1, 2, 3],
                        ticktext=["Disengaged", "Distracted", "Neutral", "Engaged"],
                    )
                )

                st.plotly_chart(fig)
        else:
            st.info(
                "No engagement data available yet. Run the Live Engagement Monitor with engagement tracking enabled."
            )

    with tab3:
        assignments_df = get_assignments_from_db()

        if not assignments_df.empty:
            st.subheader("Assignment Performance")

            # Convert submission date to datetime
            assignments_df["submission_date"] = pd.to_datetime(
                assignments_df["submission_date"]
            )

            # Filter nulls for graded assignments
            graded_df = assignments_df.dropna(subset=["grade"])

            if not graded_df.empty:
                # Student grade averages
                student_grades = graded_df.groupby("student_name")["grade"].agg(
                    ["mean", "min", "max", "count"]
                )
                student_grades.columns = [
                    "Average Grade",
                    "Lowest Grade",
                    "Highest Grade",
                    "Assignments Submitted",
                ]
                student_grades = student_grades.sort_values(
                    "Average Grade", ascending=False
                )

                st.dataframe(
                    student_grades.style.format(
                        {
                            "Average Grade": "{:.1f}",
                            "Lowest Grade": "{:.1f}",
                            "Highest Grade": "{:.1f}",
                        }
                    )
                )

                # Grade distribution
                fig = px.histogram(
                    graded_df,
                    x="grade",
                    nbins=10,
                    labels={"grade": "Grade", "count": "Number of Assignments"},
                    title="Grade Distribution",
                )
                st.plotly_chart(fig)

                # Assignment comparison
                assignment_avg = (
                    graded_df.groupby("assignment_name")["grade"].mean().sort_values()
                )

                fig = px.bar(
                    x=assignment_avg.index,
                    y=assignment_avg.values,
                    labels={"x": "Assignment", "y": "Average Grade"},
                    title="Average Grade by Assignment",
                )
                st.plotly_chart(fig)
            else:
                st.info(
                    "No graded assignments yet. Grade some assignments to see performance metrics."
                )
        else:
            st.info("No assignments submitted yet.")

elif page == "Submit Assignment":
    st.header("📤 Submit Assignment")

    col1, col2 = st.columns(2)

    with col1:
        student_name = st.selectbox("Select Student", [""] + sorted(known_face_names))
        assignment_name = st.text_input("Assignment Name")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Assignment File",
            type=["pdf", "docx", "txt", "py", "ipynb", "java", "c", "cpp"],
        )

        if (
            st.button("Submit Assignment")
            and student_name
            and assignment_name
            and uploaded_file
        ):
            # Create directory if doesn't exist
            assignment_dir = "assignments"
            if not os.path.exists(assignment_dir):
                os.makedirs(assignment_dir)

            # Save file with unique name
            file_extension = uploaded_file.name.split(".")[-1]
            unique_filename = (
                f"{student_name}_{assignment_name}_{uuid.uuid4()}.{file_extension}"
            )
            file_path = os.path.join(assignment_dir, unique_filename)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Save to database
            save_assignment_to_db(student_name, assignment_name, file_path)

            st.success(
                f"Assignment '{assignment_name}' successfully submitted for {student_name}!"
            )

    with col2:
        st.subheader("Submitted Assignments")

        assignments_df = get_assignments_from_db()

        if not assignments_df.empty:
            # Filter option
            filter_option = st.radio("Filter By", ["All", "Needs Grading", "Graded"])

            if filter_option == "Needs Grading":
                filtered_df = assignments_df[assignments_df["grade"].isna()]
            elif filter_option == "Graded":
                filtered_df = assignments_df[~assignments_df["grade"].isna()]
            else:
                filtered_df = assignments_df

            if not filtered_df.empty:
                st.dataframe(
                    filtered_df[
                        [
                            "id",
                            "student_name",
                            "assignment_name",
                            "submission_date",
                            "grade",
                        ]
                    ]
                )
            else:
                st.info(f"No assignments in '{filter_option}' category.")
        else:
            st.info("No assignments submitted yet.")

    # Teacher view for grading
    st.subheader("Grade Assignments")

    if not assignments_df.empty:
        # Create a list of assignments needing grading
        ungraded = assignments_df[assignments_df["grade"].isna()]

        if not ungraded.empty:
            selected_assignment = st.selectbox(
                "Select Assignment to Grade",
                ungraded["id"].astype(str)
                + " - "
                + ungraded["student_name"]
                + ": "
                + ungraded["assignment_name"],
            )

            if selected_assignment:
                assignment_id = int(selected_assignment.split(" - ")[0])
                assignment_row = assignments_df[
                    assignments_df["id"] == assignment_id
                ].iloc[0]

                st.write(
                    f"Grading: {assignment_row['student_name']}'s assignment '{assignment_row['assignment_name']}'"
                )

                # Provide grading inputs
                feedback = st.text_area("Feedback")
                grade = st.number_input(
                    "Grade (0-100)", min_value=0.0, max_value=100.0, step=0.5
                )

                if st.button("Submit Grade"):
                    update_assignment_feedback(assignment_id, feedback, grade)
                    st.success(
                        f"Grade and feedback submitted for {assignment_row['student_name']}!"
                    )
        else:
            st.info("All assignments have been graded!")

elif page == "Accessibility Tools":
    st.header("♿ Accessibility Enhancements")

    st.subheader("Font and Display Settings")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Text-to-Speech")
        text_to_read = st.text_area("Enter text to be read aloud:", height=150)
        voice_options = ["Default", "Female", "Male"]
        selected_voice = st.selectbox("Select Voice", voice_options)
        reading_speed = st.slider(
            "Reading Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1
        )

        if st.button("Read Text"):
            st.info("Text-to-speech functionality would play audio here.")
            st.markdown(
                f"Reading with {selected_voice} voice at {reading_speed}x speed..."
            )

    with col2:
        st.markdown("### Visual Adaptations")
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Default", "High Contrast", "Color Blind Friendly", "Dark Mode"],
        )
        font_size = st.slider("Font Size Adjustment", -50, 100, 0, 5, format="%d%%")
        enable_screen_reader = st.checkbox("Screen Reader Compatibility Mode")
        simplified_interface = st.checkbox("Simplified Interface")

        st.info("These settings would apply globally to the application interface.")

    st.subheader("Learning Accommodations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Content Adjustments")
        extra_time = st.slider(
            "Extra Time for Assessments", 0, 100, 0, 5, format="%d%%"
        )
        content_breakdown = st.checkbox("Break Content into Smaller Chunks")
        visual_aids = st.checkbox("Enhanced Visual Aids")

        if st.button("Apply Learning Accommodations"):
            st.success("Learning accommodations applied to student profile!")

    with col2:
        st.markdown("### Communication Support")
        translation_language = st.selectbox(
            "Translation Support",
            ["None", "Spanish", "French", "Chinese", "Arabic", "Hindi"],
        )

        caption_videos = st.checkbox("Auto-Caption Videos")
        sign_language = st.checkbox("Sign Language Support")

        if translation_language != "None":
            st.info(f"Content will be machine-translated to {translation_language}.")

elif page == "Student Management":
    st.header("👨‍🎓 Student Profile Management")

    tab1, tab2 = st.tabs(["Existing Students", "Add New Student"])

    with tab1:
        if known_face_names:
            selected_student = st.selectbox("Select Student", sorted(known_face_names))

            if selected_student:
                st.subheader(f"Student Profile: {selected_student}")

                col1, col2 = st.columns(2)

                with col1:
                    # Find the student's face file
                    for filename in os.listdir(FACE_DIR):
                        if os.path.splitext(filename)[0] == selected_student:
                            img_path = os.path.join(FACE_DIR, filename)
                            st.image(img_path, caption=selected_student, width=200)
                            break

                    # Attendance summary
                    attendance_df = get_attendance_from_db()
                    if not attendance_df.empty:
                        student_attendance = attendance_df[
                            attendance_df["name"] == selected_student
                        ]

                        if not student_attendance.empty:
                            st.metric("Total Days Present", len(student_attendance))

                            # Last attendance
                            last_date = pd.to_datetime(student_attendance["date"]).max()
                            st.info(
                                f"Last attended on: {last_date.strftime('%Y-%m-%d')}"
                            )
                        else:
                            st.warning("No attendance records found for this student.")

                with col2:
                    # Engagement summary if available
                    engagement_df = get_engagement_from_db()
                    if not engagement_df.empty:
                        student_engagement = engagement_df[
                            engagement_df["name"] == selected_student
                        ]

                        if not student_engagement.empty:
                            engagement_counts = student_engagement[
                                "status"
                            ].value_counts()

                            # Calculate engagement score (weighted average)
                            weights = {
                                "Engaged": 1.0,
                                "Neutral": 0.6,
                                "Distracted": 0.3,
                                "Disengaged": 0.0,
                            }

                            total_score = sum(
                                weights.get(status, 0) * count
                                for status, count in engagement_counts.items()
                            )
                            total_readings = engagement_counts.sum()

                            if total_readings > 0:
                                engagement_score = (total_score / total_readings) * 100
                                st.metric(
                                    "Engagement Score", f"{engagement_score:.1f}%"
                                )

                                # Show engagement distribution
                                st.write("Engagement Distribution:")
                                engagement_data = []
                                for status, count in engagement_counts.items():
                                    engagement_data.append(
                                        {"Status": status, "Count": count}
                                    )

                                st.dataframe(pd.DataFrame(engagement_data))
                        else:
                            st.info("No engagement data available for this student.")

                    # Assignment summary
                    assignments_df = get_assignments_from_db()
                    if not assignments_df.empty:
                        student_assignments = assignments_df[
                            assignments_df["student_name"] == selected_student
                        ]

                        if not student_assignments.empty:
                            st.write(f"Total Assignments: {len(student_assignments)}")

                            # Calculate average grade
                            graded = student_assignments.dropna(subset=["grade"])
                            if not graded.empty:
                                avg_grade = graded["grade"].mean()
                                st.metric("Average Grade", f"{avg_grade:.1f}/100")
                            else:
                                st.info("No graded assignments yet.")
                        else:
                            st.info("No assignments submitted yet.")

                # Option to delete student
                if st.button(
                    "Delete Student Profile", type="primary", use_container_width=True
                ):
                    st.warning(
                        "This will delete the student's face data and all associated records."
                    )
                    confirm = st.checkbox("I understand this action cannot be undone")

                    if confirm and st.button("Confirm Delete"):
                        # Find and delete face file
                        for filename in os.listdir(FACE_DIR):
                            if os.path.splitext(filename)[0] == selected_student:
                                try:
                                    os.remove(os.path.join(FACE_DIR, filename))
                                    st.success(
                                        f"Deleted face data for {selected_student}"
                                    )
                                    break
                                except Exception as e:
                                    st.error(f"Error deleting file: {str(e)}")

                        # Note: Database records are preserved for historical data
                        st.info(
                            "Student profile deleted. Please refresh the page to update the student list."
                        )
        else:
            st.info("No student profiles found. Add a new student to get started.")

    with tab2:
        st.subheader("Register New Student")

        new_student_name = st.text_input("Student Name")
        face_image = st.camera_input("Capture Student's Face")

        if st.button("Register Student") and new_student_name and face_image:
            # Save the captured image
            if not os.path.exists(FACE_DIR):
                os.makedirs(FACE_DIR)

            img_path = os.path.join(FACE_DIR, f"{new_student_name}.jpg")

            with open(img_path, "wb") as f:
                f.write(face_image.getbuffer())

            # Process the image to verify face is detectable
            try:
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image)

                if face_locations:
                    st.success(f"Student {new_student_name} registered successfully!")
                    st.info(
                        "Please restart the application to update the recognition database."
                    )
                else:
                    os.remove(img_path)
                    st.error(
                        "No face detected in the image. Please try again with a clearer image."
                    )
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

            # Option to upload image instead
            st.markdown("---")
            st.subheader("Or Upload an Image")
            uploaded_face = st.file_uploader(
                "Upload Student Face Image", type=["jpg", "jpeg", "png"]
            )

            if uploaded_face is not None:
                img_path = os.path.join(FACE_DIR, f"{new_student_name}.jpg")

                with open(img_path, "wb") as f:
                    f.write(uploaded_face.getbuffer())

                # Process the image to verify face is detectable
                try:
                    image = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(image)

                    if face_locations:
                        st.success(
                            f"Student {new_student_name} registered successfully!"
                        )
                        st.info(
                            "Please restart the application to update the recognition database."
                        )
                    else:
                        os.remove(img_path)
                        st.error(
                            "No face detected in the image. Please try again with a clearer image."
                        )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Release resources when switching pages
if "video_capture" in locals() and video_capture.isOpened():
    video_capture.release()
cv2.destroyAllWindows()
