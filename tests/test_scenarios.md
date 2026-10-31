# Resume Matching Agent - Conversation Test Scenarios


## Test 1: New job search
User: Find React developers with 3 years of experience.
Expected: Extract requirements, search resumes, rank candidates, deep-screen shortlist, and generate recommendations.


## Test 2: Requirement refinement
User: Also require AWS.
Expected: Updated requirements, search and rank candidates again, and regenerate screening results.


## Test 3: Candidate comparison
User: Compare the top 3 candidates.
Expected: Display a comparison of the top 3 candidates with scores, matched skills, missing skills, and explanations.


## Test 4: Ranking explanation
User: Why is Candidate A ranked higher than Candidate B?
Expected: Explain the ranking using actual match scores and skill evidence.


## Test 5: Interview Questions
User: Generate interview questions for Candidate A.
Expected: Generate relevant interview questions based on the candidate's resume and job requirements.


## Test 6: Human Feedback
User: Find React developers with 3 years of experience.
Recruiter: Review Candidate A, but do not recommend Candidate B.
Expected: Pause for recruiter feedback, resume the graph, and include feedback in the final report.