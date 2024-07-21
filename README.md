AI Apparel Graphic is an advanced AI-driven project designed to dynamically overlay logos onto images of individuals wearing various types of apparel, such as shirts, blouses, and jackets. The system precisely calculates the size, position, and angle along the y and z axes to accurately place the logos. To enhance realism, the project employs pixel-to-pixel image transformation techniques to apply wrinkles to the imposed logos. Additionally, the project allows for the alteration of shirt colors using the HSV color space.

The project integrates several sophisticated technologies and models:

K-Nearest Neighbors (KNN): Utilized to classify shirt and non-shirt regions.
'segformer-b2-fashio': An open-source model employed for segmenting different outfit regions.
Google MediaPipe: Used for detecting body key points, such as shoulders, to ensure precise logo placement.

Overall, AI Apparel Graphic represents a comprehensive AI and computer vision solution for realistic and dynamic logo imposition on apparel images.

How to execute project : 
It is a streamlit app, just activate the virtual environment, run requirements.txt ' pip install -r requirements. txt' and than run command 'streamlit run app.py'
