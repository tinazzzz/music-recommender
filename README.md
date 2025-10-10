# music-recommender
A recommendation system for music

-----------------------------
Setup
-----------------------------

1. Download HetRec 2011 Last.fm dataset and place under data/raw/hetrec2011-lastfm-2k/



-----------------------------
Flow
-----------------------------

1. Pick whether you want to see recommendations for existing users / add yourself as a new user
2. existing -> hybrid (MF + semantic)
3. new  -> fall back to pure semantic on the next 3-5 recommendations
        -> get feedback from the 3-5 recommendations
        -> get temp MF embeddings using user embedding projection
        -> hybrid recommendation (MF + semantic) after 3-5

                  ┌────────────────────┐
                  │ User enters system │
                  └──────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │ Existing User   │           │ New User        │
     └───────┬─────────┘           └────────┬────────┘
             │                              │
   ┌─────────▼────────┐        ┌────────────▼────────────┐
   │ Hybrid (MF+Sem.) │        │ Semantic-only onboarding │
   └─────────┬────────┘        └────────────┬────────────┘
             │                              │
             │                  ┌───────────▼──────────────┐
             │                  │ Collect 3–5 interactions │
             │                  └───────────┬──────────────┘
             │                              │
             │                  ┌───────────▼──────────────┐
             │                  │ Project temp MF embedding│
             │                  └───────────┬──────────────┘
             │                              │
             └──────────────► Hybrid (MF + Semantic)

