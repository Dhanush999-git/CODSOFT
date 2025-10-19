# 🎮 Task 2: Tic Tac Toe AI

## Project Overview
This project is part of the CODSOFT Artificial Intelligence Internship. In this task, I built an **unbeatable Tic Tac Toe game** using the Minimax algorithm with Alpha-Beta pruning. The AI evaluates all possible moves to ensure it never loses, providing a challenging experience for the player. This project helps in understanding **game theory, search algorithms, and AI decision-making**.

## Features
- Play Tic Tac Toe against an AI  
- AI uses **Minimax algorithm with Alpha-Beta pruning** for optimal moves  
- User can choose to play first (X) or second (O)  
- Shows numbered board positions for easy input  
- Handles quitting gracefully (q, quit, exit)  
- Draw detection when the board is full  

## Technologies Used
- Python 3  
- Standard libraries: random, math, sys

## Project Structure
Task2_tictactoe/                                                                                                                           
│── tictactoe_ai.py # Main game script                                                                                                                         
│── README.md # Documentation (this file)                                                                                                                         
│── requirements.txt # Dependencies  

## How to Run
1. Clone this repository or download the folder.  
2. Open a terminal and navigate to Task2_tictactoe/.  
3. Run the game:  
   python tictactoe_ai.py

## Example Interaction
Tic Tac Toe Game
Play as X (first) or O (second)? [X]: X

 1 | 2 | 3                                                       
---+---+---                                                   
 4 | 5 | 6                                                        
---+---+---                                                       
 7 | 8 | 9                                                                     

Enter move (1-9) or q to quit: 1
AI is thinking...

 X | 2 | 3                                                   
---+---+---                                                          
 4 | O | 6                                                            
---+---+---                                                           
 7 | 8 | 9                                                           

## Future Improvements
- Add difficulty levels (Easy: random moves, Medium: depth-limited Minimax)
- Highlight the winning line when the game ends
- Implement a GUI version using tkinter or pygame
- Keep track of player statistics and scores
- Expand to multiplayer or networked play
