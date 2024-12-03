uvicorn embedding:app --host 0.0.0.0 --port 9502




curl -X POST "http://0.0.0.0:9502/similarity" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is the cycle life of this 3.2V 280ah Lifepo4 battery?",
  "candidates": [
    "https://sc04.alicdn.com/kf/H3510328463d740b2afbcf401c8c108f2J/240062176/H3510328463d740b2afbcf401c8c108f2J.jpg",
    "https://sc04.alicdn.com/kf/H75608c12162a47a4ad41fd331c212e29X/240062176/H75608c12162a47a4ad41fd331c212e29X.jpg",
    "https://sc04.alicdn.com/kf/H1c593aa026e64725a43e1a538be6951ay/240062176/H1c593aa026e64725a43e1a538be6951ay.jpg",
    "https://sc04.alicdn.com/kf/Hb7cda33e8bdc476091ff2962cb4f0ae3x/240062176/Hb7cda33e8bdc476091ff2962cb4f0ae3x.jpg",
    "https://sc04.alicdn.com/kf/Hc00c90da8dcb43b8aeee7eb11b12291b1/240062176/Hc00c90da8dcb43b8aeee7eb11b12291b1.jpg",
    "https://sc04.alicdn.com/kf/H9b13be1329344c3a96f295f144932582u/240062176/H9b13be1329344c3a96f295f144932582u.jpg",
    "https://sc04.alicdn.com/kf/H471ca5edf21a4caea852192af7fbefe7T/240062176/H471ca5edf21a4caea852192af7fbefe7T.jpg",
    "https://sc04.alicdn.com/kf/H38de8263ae5847cb9e6662cdee53743cA/240062176/H38de8263ae5847cb9e6662cdee53743cA.jpg",
    "https://sc04.alicdn.com/kf/H1ea2aa793f5c4d009923d18a473ac219k/240062176/H1ea2aa793f5c4d009923d18a473ac219k.png",
    "https://sc04.alicdn.com/kf/H7fec7cd6293c48168fdd1d41c48ab9e0O/240062176/H7fec7cd6293c48168fdd1d41c48ab9e0O.jpg",
    "https://sc04.alicdn.com/kf/He8d4b88d4323492689455acfa3e44564g/240062176/He8d4b88d4323492689455acfa3e44564g.jpg",
    "https://sc04.alicdn.com/kf/Hff4f46cf682d4deea2094bb71ecc446fu/240062176/Hff4f46cf682d4deea2094bb71ecc446fu.jpg",
    "https://sc04.alicdn.com/kf/Hc5b49b124f1c491aa2fb3078a921929db/240062176/Hc5b49b124f1c491aa2fb3078a921929db.png"
  ],
  "query_type": "text",
  "candidate_type": "image"
}'



curl -X POST "http://0.0.0.0:9502/similarity" \
-H "Content-Type: application/json" \
-d '{
  "query": "How old are you?",
  "candidates": [
    "what is your age?",
    "How are you?",
    "Hello, how tall are you?"
  ],
  "query_type": "text",
  "candidate_type": "text"
}'