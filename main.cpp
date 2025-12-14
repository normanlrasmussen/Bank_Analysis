#include <cstdint>
#include <iostream>
#include <vector>
#include <sstream>
#include <random>
#include <thread>
#include <unordered_map>
#include <chrono>

// Game constancts
static const int NUM_ROUNDS = 5;
static const int MAX_DEFICIT = 600;
static const int MAX_ROUND_SCORE = 300;
static const int OPT_OUT_POINTS = 100;
static const double WIN_WEIGHT = 0.75;

// Define decision point enum
enum DecisionPoint
{
    OpponentInitOptOut,
    ComputerInitOptOut,
    OpponentOptOut,
    ComputerOptOut,
    FirstChance,
    Chance
};

// Define game over enum
enum GameOver
{
    NotOver,
    Win,
    Loss,
    Tie
};

class GameState
{
private:
    int round = 1;
    int score_deficit = 0;
    bool computer_first = true;
    bool computer_opt_out = false;
    bool opponent_opt_out = false;
    int round_score = false;
    DecisionPoint decision_point = OpponentInitOptOut;
    GameOver game_over = NotOver;
    size_t h;

    void set_hash_value()
    {
        std::hash<int> hash_int;
        std::hash<bool> hash_bool;

        h = hash_int(round);
        h = h * 31 + hash_int(score_deficit);
        h = h * 31 + hash_bool(computer_first);
        h = h * 31 + hash_bool(computer_opt_out);
        h = h * 31 + hash_bool(opponent_opt_out);
        h = h * 31 + hash_int(round_score);
        h = h * 31 + hash_int(decision_point);
        h = h * 31 + hash_int(game_over);
    }

public:
    GameState(int round, int score_deficit, bool computer_first, bool computer_opt_out,
              bool opponent_opt_out, int round_score, DecisionPoint decision_point)
    {
        this->round = round;
        if (score_deficit < (-MAX_DEFICIT))
        {
            this->score_deficit = -MAX_DEFICIT;
        }
        else if (score_deficit > MAX_DEFICIT)
        {
            this->score_deficit = MAX_DEFICIT;
        }
        else
        {
            this->score_deficit = score_deficit;
        }
        this->computer_first = computer_first;
        this->computer_opt_out = computer_opt_out;
        this->opponent_opt_out = opponent_opt_out;
        if (round_score < (-MAX_ROUND_SCORE))
        {
            this->round_score = -MAX_ROUND_SCORE;
        }
        else if (round_score > MAX_ROUND_SCORE)
        {
            this->round_score = MAX_ROUND_SCORE;
        }
        else
        {
            this->round_score = round_score;
        }
        this->decision_point = decision_point;
        this->game_over = NotOver;
        this->set_hash_value();
    }

    size_t get_hash_value() const
    {
        return h;
    }

    DecisionPoint get_decision_point() const
    {
        return decision_point;
    }

    GameState get_reverse() const
    {
        if (game_over != NotOver)
        {
            return GameState(game_over == Win ? Loss : game_over == Loss ? Win
                                                                         : Tie);
        }
        else
        {
            DecisionPoint reverse_decision_point = decision_point;
            if (decision_point == ComputerInitOptOut)
            {
                reverse_decision_point = OpponentInitOptOut;
            }
            else if (decision_point == OpponentInitOptOut)
            {
                reverse_decision_point = ComputerInitOptOut;
            }
            else if (decision_point == ComputerOptOut)
            {
                reverse_decision_point = OpponentOptOut;
            }
            else if (decision_point == OpponentOptOut)
            {
                reverse_decision_point = ComputerOptOut;
            }
            return GameState(round, -score_deficit, !computer_first, opponent_opt_out, computer_opt_out, round_score, reverse_decision_point);
        }
    }

    GameState(int score_deficit)
    {
        if (score_deficit < (-MAX_DEFICIT))
        {
            this->score_deficit = -MAX_DEFICIT;
        }
        else if (score_deficit > MAX_DEFICIT)
        {
            this->score_deficit = MAX_DEFICIT;
        }
        else
        {
            this->score_deficit = score_deficit;
        }
        this->game_over = (score_deficit > 0) ? Win : (score_deficit < 0) ? Loss
                                                                          : Tie;
        this->set_hash_value();
    }

    bool is_over() const
    {
        return this->game_over != NotOver;
    }

    void get_transitions(DecisionPoint &decision_point, std::vector<GameState> &transitions)
    {
        // Set the decision point
        decision_point = this->decision_point;
        // Check the decision point
        switch (decision_point)
        {
        case ComputerInitOptOut:
        {
            transitions.push_back(GameState(round, score_deficit, computer_first, false, opponent_opt_out, round_score, computer_first ? OpponentInitOptOut : FirstChance));
            transitions.push_back(GameState(round, score_deficit, computer_first, true, opponent_opt_out, round_score, computer_first ? OpponentInitOptOut : FirstChance));
        }
        break;
        case OpponentInitOptOut:
        {
            transitions.push_back(GameState(round, score_deficit, computer_first, computer_opt_out, false, round_score, !computer_first ? ComputerInitOptOut : FirstChance));
            transitions.push_back(GameState(round, score_deficit, computer_first, computer_opt_out, true, round_score, !computer_first ? ComputerInitOptOut : FirstChance));
        }
        break;
        case FirstChance:
        {
            // Add options for each roll
            for (int roll : {1, 2, 3, 4, 5, 6})
            {
                if (roll == 1)
                {
                    int new_score_deficit = score_deficit + ((computer_opt_out && !opponent_opt_out) ? OPT_OUT_POINTS : (!computer_opt_out && opponent_opt_out) ? (-OPT_OUT_POINTS)
                                                                                                                                                                : 0);
                    if (round == NUM_ROUNDS)
                    {
                        transitions.push_back(GameState(new_score_deficit));
                    }
                    else
                    {
                        transitions.push_back(GameState(round + 1, new_score_deficit, !computer_first, false, false, 0, computer_first ? OpponentInitOptOut : ComputerInitOptOut));
                    }
                }
                else
                {
                    if (computer_opt_out && opponent_opt_out)
                    {
                        if (round == NUM_ROUNDS)
                        {
                            transitions.push_back(GameState(score_deficit));
                        }
                        else
                        {
                            transitions.push_back(GameState(round + 1, score_deficit, !computer_first, false, false, 0, computer_first ? OpponentInitOptOut : ComputerInitOptOut));
                        }
                    }
                    else
                    {
                        DecisionPoint next_decision_point = ((computer_first && !computer_opt_out) || opponent_opt_out) ? ComputerOptOut : OpponentOptOut;
                        transitions.push_back(GameState(round, score_deficit, computer_first, computer_opt_out, opponent_opt_out, roll, next_decision_point));
                    }
                }
            }
        }
        break;
        case ComputerOptOut:
        {
            transitions.push_back(GameState(round, score_deficit, computer_first, false, opponent_opt_out, round_score, (computer_first && !opponent_opt_out) ? OpponentOptOut : Chance));
            int new_score_deficit = score_deficit + round_score;
            if (!opponent_opt_out)
            {
                transitions.push_back(GameState(round, new_score_deficit, computer_first, true, opponent_opt_out, round_score, computer_first ? OpponentOptOut : Chance));
            }
            else if (round == NUM_ROUNDS)
            {
                transitions.push_back(GameState(new_score_deficit));
            }
            else
            {
                transitions.push_back(GameState(round + 1, new_score_deficit, !computer_first, false, false, 0, computer_first ? OpponentInitOptOut : ComputerInitOptOut));
            }
        }
        break;
        case OpponentOptOut:
        {
            transitions.push_back(GameState(round, score_deficit, computer_first, computer_opt_out, false, round_score, (!computer_first && !computer_opt_out) ? ComputerOptOut : Chance));
            int new_score_deficit = score_deficit - round_score;
            if (!computer_opt_out)
            {
                transitions.push_back(GameState(round, new_score_deficit, computer_first, computer_opt_out, true, round_score, !computer_first ? ComputerOptOut : Chance));
            }
            else if (round == NUM_ROUNDS)
            {
                transitions.push_back(GameState(new_score_deficit));
            }
            else
            {
                transitions.push_back(GameState(round + 1, new_score_deficit, !computer_first, false, false, 0, computer_first ? OpponentInitOptOut : ComputerInitOptOut));
            }
        }
        break;
        case Chance:
        {
            // Add options for each roll
            for (int roll : {1, 2, 3, 4, 5, 6})
            {
                if (roll == 1)
                {
                    if (round == NUM_ROUNDS)
                    {
                        transitions.push_back(GameState(score_deficit));
                    }
                    else
                    {
                        transitions.push_back(GameState(round + 1, score_deficit, !computer_first, false, false, 0, computer_first ? OpponentInitOptOut : ComputerInitOptOut));
                    }
                }
                else
                {
                    int new_round_score = (roll == 2) ? (2 * round_score) : (roll + round_score);
                    if (new_round_score >= MAX_ROUND_SCORE)
                    {
                        int new_score_deficit = score_deficit + ((!computer_opt_out && opponent_opt_out) ? new_round_score : ((computer_opt_out && !opponent_opt_out) ? -new_round_score : 0));
                        if (round == NUM_ROUNDS)
                        {
                            transitions.push_back(GameState(new_score_deficit));
                        }
                        else
                        {
                            transitions.push_back(GameState(round + 1, new_score_deficit, !computer_first, false, false, 0, computer_first ? OpponentInitOptOut : ComputerInitOptOut));
                        }
                    }
                    else
                    {
                        DecisionPoint next_decision_point = ((computer_first && !computer_opt_out) || opponent_opt_out) ? ComputerOptOut : OpponentOptOut;
                        transitions.push_back(GameState(round, score_deficit, computer_first, computer_opt_out, opponent_opt_out, new_round_score, next_decision_point));
                    }
                }
            }
        }
        break;
        }
    }

    // Override the equal operator
    bool operator==(const GameState &other) const
    {
        return round == other.round &&
               score_deficit == other.score_deficit &&
               computer_first == other.computer_first &&
               computer_opt_out == other.computer_opt_out &&
               opponent_opt_out == other.opponent_opt_out &&
               round_score == other.round_score &&
               decision_point == other.decision_point &&
               game_over == other.game_over;
    }

    std::string to_string() const
    {
        if (game_over != NotOver)
        {
            if (game_over == Win)
            {
                return "Computer wins!";
            }
            else if (game_over == Loss)
            {
                return "You win!";
            }
            else
            {
                return "It's a tie!";
            }
        }

        std::stringstream ss;
        ss << "Round: " << this->round << std::endl;
        ss << "Score Deficit: " << score_deficit << std::endl;
        ss << "Round Score: " << round_score << std::endl;
        ss << "Computer First: " << computer_first << std::endl;
        ss << "Computer Opt Out: " << computer_opt_out << std::endl;
        ss << "User Opt Out: " << opponent_opt_out << std::endl;
        switch (decision_point)
        {
        case OpponentInitOptOut:
            ss << "Decision Point: OpponentInitOptOut";
            break;
        case ComputerInitOptOut:
            ss << "Decision Point: ComputerInitOptOut";
            break;
        case OpponentOptOut:
            ss << "Decision Point: OpponentOptOut";
            break;
        case ComputerOptOut:
            ss << "Decision Point: ComputerOptOut";
            break;
        case FirstChance:
            ss << "Decision Point: FirstChance";
            break;
        case Chance:
            ss << "Decision Point: Chance";
            break;
        }
        return ss.str();
    }

    // Override << operator
    friend std::ostream &operator<<(std::ostream &os, const GameState &game_state)
    {
        os << game_state.to_string();
        return os;
    }
};

struct GameStateHash
{
    size_t operator()(const GameState &obj) const
    {
        return obj.get_hash_value();
    }
};

static std::unordered_map<GameState, std::pair<std::pair<double, double>, int>, GameStateHash> ALL_STATES;

std::pair<double, double> get_win_prob(GameState game_state)
{
    std::vector<GameState> transitions;
    DecisionPoint decision_point;
    game_state.get_transitions(decision_point, transitions);
    double computer_win_value = 0.0;
    double opponent_win_value = 0.0;
    if ((decision_point == FirstChance) || (decision_point == Chance))
    {
        for (GameState transition : transitions)
        {
            if (ALL_STATES.find(transition) != ALL_STATES.end())
            {
                computer_win_value += ALL_STATES.at(transition).first.first;
                opponent_win_value += ALL_STATES.at(transition).first.second;
            }
            else
            {
                auto values = get_win_prob(transition);
                computer_win_value += values.first;
                opponent_win_value += values.second;
            }
        }
        computer_win_value /= 6;
        opponent_win_value /= 6;
        ALL_STATES[game_state] = std::make_pair(std::make_pair(computer_win_value, opponent_win_value), 0);
    }
    else if ((decision_point == ComputerInitOptOut) || (decision_point == ComputerOptOut))
    {
        computer_win_value = -1.0;
        int choice = 0;
        for (int index = 0; index < transitions.size(); index++)
        {
            std::pair<double, double> choice_values;
            if (ALL_STATES.find(transitions.at(index)) != ALL_STATES.end())
            {
                choice_values = ALL_STATES.at(transitions.at(index)).first;
            }
            else
            {
                choice_values = get_win_prob(transitions.at(index));
            }
            if (choice_values.first > computer_win_value)
            {
                computer_win_value = choice_values.first;
                opponent_win_value = choice_values.second;
                choice = index;
            }
        }
        ALL_STATES[game_state] = std::make_pair(std::make_pair(computer_win_value, opponent_win_value), choice);
    }
    else
    {
        opponent_win_value = -1.0;
        int choice = 0;
        for (int index = 0; index < transitions.size(); index++)
        {
            std::pair<double, double> choice_values;
            if (ALL_STATES.find(transitions.at(index)) != ALL_STATES.end())
            {
                choice_values = ALL_STATES.at(transitions.at(index)).first;
            }
            else
            {
                choice_values = get_win_prob(transitions.at(index));
            }
            if (choice_values.second > opponent_win_value)
            {
                computer_win_value = choice_values.first;
                opponent_win_value = choice_values.second;
                choice = index;
            }
        }
        ALL_STATES[game_state] = std::make_pair(std::make_pair(computer_win_value, opponent_win_value), choice);
    }
    return std::make_pair(computer_win_value, opponent_win_value);
}

int main()
{
    for (int score_deficit = -MAX_DEFICIT; score_deficit < MAX_DEFICIT + 1; score_deficit++)
    {
        double computer_win_value = (score_deficit > 0) ? 1.0 : 0;
        double opponent_win_value = (score_deficit < 0) ? 1.0 : 0;
        double computer_point_value = (score_deficit > 0) ? (score_deficit / MAX_DEFICIT) : 0;
        double opponent_point_value = (score_deficit < 0) ? (-score_deficit / MAX_DEFICIT) : 0;
        double computer_value = WIN_WEIGHT * computer_win_value + (1 - WIN_WEIGHT) * computer_point_value;
        double opponent_value = WIN_WEIGHT * opponent_win_value + (1 - WIN_WEIGHT) * opponent_point_value;
        ALL_STATES[GameState(score_deficit)] = std::make_pair(std::make_pair(computer_value, opponent_value), 0);
    }

    GameState computer_first_init = GameState(1, 0, true, false, false, 0, ComputerInitOptOut);
    GameState opponent_first_init = GameState(1, 0, false, false, false, 0, OpponentInitOptOut);

    std::pair<double, double> values = get_win_prob(computer_first_init);
    std::cout << values.first << std::endl;
    values = get_win_prob(opponent_first_init);
    std::cout << values.first << std::endl;

    bool keep_playing = true;
    bool show_intructions = false;

    while (keep_playing)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<> dis(0, 1);
        bool computer_first;
        if (dis(gen) == 1)
        {
            computer_first = false;
            std::cout << "User goes first\n\n";
        }
        else
        {
            computer_first = true;
            std::cout << "Computer goes first\n\n";
        }

        int pause_time = 500;

        int round = 1;
        int computer_score = 0;
        int user_score = 0;
        while (round <= NUM_ROUNDS)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
            std::cout << "Round: " << round << "\n";
            std::cout << "Computer: " << computer_score << "\n";
            std::cout << "User: " << user_score << "\n";
            std::cout << (computer_first ? "Computer goes first\n\n" : "User goes first\n\n");
            int round_score = 0;
            bool computer_opt_out = false;
            bool user_opt_out = false;
            bool is_first = true;
            while (!computer_opt_out || !user_opt_out)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
                std::cout << "Round: " << round << "\n";
                std::cout << "Computer: " << computer_score << (computer_opt_out ? " (opted out)\n" : "\n");
                std::cout << "User: " << user_score << (user_opt_out ? " (opted out)\n" : "\n");
                std::cout << "Round Score: " << round_score << "\n";
                DecisionPoint next_decision;
                if (is_first)
                {
                    next_decision = computer_first ? ComputerInitOptOut : OpponentInitOptOut;
                }
                else
                {
                    if (computer_first)
                    {
                        next_decision = computer_opt_out ? OpponentOptOut : ComputerOptOut;
                    }
                    else
                    {
                        next_decision = user_opt_out ? ComputerOptOut : OpponentOptOut;
                    }
                }
                auto values = ALL_STATES.at(GameState(round, computer_score - user_score, computer_first,
                                                      computer_opt_out, user_opt_out, round_score, next_decision))
                                  .first;
                std::cout << "User odds: " << values.second << ", Computer odds: " << values.first << "\n\n";
                // Give opportunities to opt out
                if (computer_first)
                {
                    if (!computer_opt_out)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
                        GameState game_state = GameState(round, computer_score - user_score, computer_first, computer_opt_out,
                                                         user_opt_out, round_score, is_first ? ComputerInitOptOut : ComputerOptOut);
                        computer_opt_out = ALL_STATES.at(game_state).second;
                        if (computer_opt_out)
                        {
                            computer_score += round_score;
                            std::cout << "Computer has opted out\n\n";
                        }
                        else
                        {
                            std::cout << "Computer will continue rolling\n\n";
                        }
                    }
                    if (!user_opt_out)
                    {
                        std::cout << "Press Enter to continue rolling, anything else to opt out\n";
                        GameState game_state = GameState(round, computer_score - user_score, computer_first, computer_opt_out,
                                                         user_opt_out, round_score, is_first ? OpponentInitOptOut : OpponentOptOut);
                        int recommendation = ALL_STATES.at(game_state).second;
                        if (show_intructions)
                        {
                            std::cout << "Recommendation:" << (recommendation ? " opt out\n" : " continue\n");
                        }
                        std::string response;
                        std::getline(std::cin, response);
                        if (response == "")
                        {
                            std::cout << "User will continue rolling\n\n";
                        }
                        else if (response == "open")
                        {
                            show_intructions = true;
                            std::cout << "Unlocked recommendations\nUser will continue rolling\n\n";
                        }
                        else if (response == "close")
                        {
                            show_intructions = false;
                            std::cout << "Locked recommendations\nUser will continue rolling\n\n";
                        }
                        else
                        {
                            user_opt_out = true;
                            user_score += round_score;
                            std::cout << "User has opted out\n\n";
                        }
                    }
                }
                else
                {
                    if (!user_opt_out)
                    {
                        std::cout << "Press Enter to continue rolling, anything else to opt out\n";
                        GameState game_state = GameState(round, computer_score - user_score, computer_first, computer_opt_out,
                                                         user_opt_out, round_score, is_first ? OpponentInitOptOut : OpponentOptOut);
                        int recommendation = ALL_STATES.at(game_state).second;
                        if (show_intructions)
                        {
                            std::cout << "Recommendation:" << (recommendation ? " opt out\n" : " continue\n");
                        }
                        std::string response;
                        std::getline(std::cin, response);
                        if (response == "")
                        {
                            std::cout << "User will continue rolling\n\n";
                        }
                        else if (response == "open")
                        {
                            show_intructions = true;
                            std::cout << "Unlocked recommendations\nUser will continue rolling\n\n";
                        }
                        else if (response == "close")
                        {
                            show_intructions = false;
                            std::cout << "Locked recommendations\nUser will continue rolling\n\n";
                        }
                        else
                        {
                            user_opt_out = true;
                            user_score += round_score;
                            std::cout << "User has opted out\n\n";
                        }
                    }
                    if (!computer_opt_out)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
                        GameState game_state = GameState(round, computer_score - user_score, computer_first, computer_opt_out,
                                                         user_opt_out, round_score, is_first ? ComputerInitOptOut : ComputerOptOut);
                        computer_opt_out = ALL_STATES.at(game_state).second;
                        if (computer_opt_out)
                        {
                            computer_score += round_score;
                            std::cout << "Computer has opted out\n\n";
                        }
                        else
                        {
                            std::cout << "Computer will continue rolling\n\n";
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
                std::cout << "Computer: " << computer_score << (computer_opt_out ? " (opted out)\n" : "\n");
                std::cout << "User: " << user_score << (user_opt_out ? " (opted out)\n" : "\n");
                std::cout << "Round Score: " << round_score << "\n\n";
                // Roll die
                std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
                std::uniform_int_distribution<> dis(1, 6);
                int roll = dis(gen);
                std::cout << "Roll is " << roll << "\n\n";
                // Check if is the first
                if (is_first)
                {
                    is_first = false;
                    if (computer_opt_out)
                    {
                        computer_score += ((roll == 1) ? OPT_OUT_POINTS : 0);
                    }
                    if (user_opt_out)
                    {
                        user_score += ((roll == 1) ? OPT_OUT_POINTS : 0);
                    }
                    if (roll > 1)
                    {
                        round_score = roll;
                    }
                    else
                    {
                        computer_opt_out = true;
                        user_opt_out = true;
                    }
                }
                else
                {
                    if (roll == 1)
                    {
                        computer_opt_out = true;
                        user_opt_out = true;
                    }
                    else if (roll == 2)
                    {
                        round_score *= 2;
                    }
                    else
                    {
                        round_score += roll;
                    }
                }
            }
            round++;
            computer_first = !computer_first;
        }

        if (computer_score > user_score)
        {
            std::cout << "Computer wins!\n\n\n";
        }
        else if (user_score > computer_score)
        {
            std::cout << "User wins!\n\n\n";
        }
        else
        {
            std::cout << "It's a tie!\n\n\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(pause_time));
        std::cout << "Press Enter to keep play, anything else to quit\n";
        std::string response;
        std::getline(std::cin, response);
        if (response != "")
        {
            keep_playing = false;
        }
        else
        {
            std::cout << "New Game!\n\n";
        }
    }

    return 0;
}
