# COMMAND ----------

# import statements for the entire notebook
# add anything that is required here

import re
from typing import Dict, List, Tuple
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, ArrayType, BooleanType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from collections import Counter

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 1 - Video game sales data
# MAGIC
# MAGIC The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container] contains video game sales data (based on [https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom](https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom)).
# MAGIC
# MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
# MAGIC
# MAGIC Using the data, find answers to the following:
# MAGIC
# MAGIC - Which publisher has the highest total sales in video games in North America considering games released in years 2006-2015?
# MAGIC - How many titles in total for this publisher do not have sales data available for North America considering games released in years 2006-2015?
# MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2006-2015, what are the total sales, in North America and globally, for each year?
# MAGIC     - I.e., what are the total sales (in North America and globally) for games released by this publisher in year 2006? And the same for year 2007? ...
# MAGIC

# COMMAND ----------

game_sales_path: str = ".../assignment/sales/video_game_sales.csv"

# Define the schema for the video game sales
schema = StructType([
    StructField("title", StringType(), True),
    StructField("publisher", StringType(), True),
    StructField("developer", StringType(), True),
    StructField("release_date", DateType(), True),
    StructField("platform", StringType(), True),
    StructField("total_sales", DoubleType(), True),
    StructField("na_sales", DoubleType(), True),
    StructField("japan_sales", DoubleType(), True),
    StructField("pal_sales", DoubleType(), True),
    StructField("other_sales", DoubleType(), True),
    StructField("user_score", DoubleType(), True),
    StructField("critic_score", DoubleType(), True),
])
salesDF = spark.read.schema(schema).csv(game_sales_path, header=True, sep="|")

# Select only needed columns early
salesDF = salesDF.select("publisher", "na_sales", "total_sales", "release_date")

# Add a new 'year' column extracted from 'release_date'
salesDF: DataFrame = salesDF.withColumn("year", F.year(F.to_date(F.col("release_date"), "dd-MM-yyyy")))

# Filter data to consider games released between 2006 and 2015
salesDF_filtered: DataFrame = salesDF.filter((F.col("year") >= 2006) & (F.col("year") <= 2015)).cache()

# Find publisher with highest NA sales in this period
salesDF_na_total: DataFrame = salesDF_filtered.groupBy("publisher").agg(F.sum("na_sales").alias("na_total"))
bestNAPublisher: str = salesDF_na_total.sort(F.desc("na_total")).limit(1).select("publisher").collect()[0][0]

# Count the titles with missing NA sales for this publisher
titlesWithMissingSalesData: str = salesDF_filtered.filter(
    (F.col("publisher") == bestNAPublisher) & (F.col("na_sales").isNull())
).count()

# Group by year to calculate total NA and global sales
bestNAPublisherSales: DataFrame = salesDF_filtered.filter(F.col("publisher") == bestNAPublisher).groupBy("year") \
.agg(
    F.round(F.sum("na_sales"), 2).alias("na_sales_by_year"),
    F.round(F.sum("total_sales"), 2).alias("global_sales_by_year")
).orderBy("year")

print(f"The publisher with the highest total video game sales in North America is: '{bestNAPublisher}'")
print(f"The number of titles with missing sales data for North America: {titlesWithMissingSalesData}")
print("Sales data for the publisher:")
bestNAPublisherSales.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 2 - Event data from football matches
# MAGIC
# MAGIC A parquet file in the [Shared container] at folder `assignment/football/events.parquet` based on [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1.
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
# MAGIC
# MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
# MAGIC
# MAGIC The team that scores more goals than their opponent wins the match.
# MAGIC
# MAGIC **Columns in the data**
# MAGIC
# MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.
# MAGIC
# MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string | The name of the competition |
# MAGIC | season | string | The season the match was played |
# MAGIC | matchId | integer | A unique id for the match |
# MAGIC | eventId | integer | A unique id for the event |
# MAGIC | homeTeam | string | The name of the home team |
# MAGIC | awayTeam | string | The name of the away team |
# MAGIC | event | string | The main category for the event |
# MAGIC | subEvent | string | The subcategory for the event |
# MAGIC | eventTeam | string | The name of the team that initiated the event |
# MAGIC | eventPlayerId | integer | The id for the player who initiated the event |
# MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
# MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
# MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
# MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC
# MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
# MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
# MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
# MAGIC
# MAGIC #### The task
# MAGIC
# MAGIC In this task you should load the data with all the rows into a data frame. This data frame object will then be used in the following basic tasks 3-8.

# COMMAND ----------

football_path: str = ".../assignment/football/events.parquet"
eventDF: DataFrame = spark.read.parquet(football_path)

eventDF = eventDF.drop("eventPlayerId", "eventTime", "startPosition", "endPosition", "eventPeriod")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 3 - Calculate match results
# MAGIC
# MAGIC Create a match data frame for all the matches included in the event data frame created in basic task 2.
# MAGIC
# MAGIC The resulting data frame should contain one row for each match and include the following columns:
# MAGIC
# MAGIC | column name   | column type | description |
# MAGIC | ------------- | ----------- | ----------- |
# MAGIC | matchId       | integer     | A unique id for the match |
# MAGIC | competition   | string      | The name of the competition |
# MAGIC | season        | string      | The season the match was played |
# MAGIC | homeTeam      | string      | The name of the home team |
# MAGIC | awayTeam      | string      | The name of the away team |
# MAGIC | homeTeamGoals | integer     | The number of goals scored by the home team |
# MAGIC | awayTeamGoals | integer     | The number of goals scored by the away team |
# MAGIC
# MAGIC The number of goals scored for each team should be determined by the available event data.<br>
# MAGIC There are two events related to each goal:
# MAGIC
# MAGIC - One event for the player that scored the goal. This includes possible own goals.
# MAGIC - One event for the goalkeeper that tried to stop the goal.
# MAGIC
# MAGIC You need to choose which types of events you are counting.<br>
# MAGIC If you count both of the event types mentioned above, you will get double the amount of actual goals.

# COMMAND ----------

goalEventsDF: DataFrame = eventDF.filter(
    (F.col("event") == "Save attempt") & 
    (F.array_contains(F.col("tags"), "Goal"))
)

# Grouping by match details and summing goals
goalCountsDF = goalEventsDF.groupBy("matchId","competition", "season", "homeTeam", "awayTeam").agg(
    F.sum(F.when(F.col("awayTeam") == F.col("eventTeam"), 1).otherwise(0)).cast(IntegerType()).alias("homeTeamGoals"),
    F.sum(F.when(F.col("homeTeam") == F.col("eventTeam"), 1).otherwise(0)).cast(IntegerType()).alias("awayTeamGoals")
)

# Create a DataFrame that includes all matches, even those with no goals
allMatchesDF = eventDF.select("matchId", "competition", "season", "homeTeam", "awayTeam").distinct()

# Join the goal counts with all matches to include matches with no goals
matchDF = allMatchesDF.join(goalCountsDF, 
                            on=["matchId", "competition", "season", "homeTeam", "awayTeam"], 
                            how="left")

# Fill missing goal values with 0 (for matches that ended 0-0)
matchDF = matchDF.fillna({"homeTeamGoals": 0, "awayTeamGoals": 0})


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 4 - Calculate team points in a season
# MAGIC
# MAGIC Create a season data frame that uses the match data frame from the basic task 3 and contains aggregated seasonal results and statistics for all the teams in all leagues. While the used dataset only includes data from a single season for each league, the code should be written such that it would work even if the data would include matches from multiple seasons for each league.
# MAGIC
# MAGIC ###### Game result determination
# MAGIC
# MAGIC - Team wins the match if they score more goals than their opponent.
# MAGIC - The match is considered a draw if both teams score equal amount of goals.
# MAGIC - Team loses the match if they score fewer goals than their opponent.
# MAGIC
# MAGIC ###### Match point determination
# MAGIC
# MAGIC - The winning team gains 3 points from the match.
# MAGIC - Both teams gain 1 point from a drawn match.
# MAGIC - The losing team does not gain any points from the match.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team per league and season. It should include the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | competition    | string      | The name of the competition |
# MAGIC | season         | string      | The season |
# MAGIC | team           | string      | The name of the team |
# MAGIC | games          | integer     | The number of games the team played in the given season |
# MAGIC | wins           | integer     | The number of wins the team had in the given season |
# MAGIC | draws          | integer     | The number of draws the team had in the given season |
# MAGIC | losses         | integer     | The number of losses the team had in the given season |
# MAGIC | goalsScored    | integer     | The total number of goals the team scored in the given season |
# MAGIC | goalsConceded  | integer     | The total number of goals scored against the team in the given season |
# MAGIC | points         | integer     | The total number of points gained by the team in the given season |

# COMMAND ----------

# Step 1: Create teamsDF with proper goalsScored and goalsConceded
teamsDF = matchDF.withColumn("team", F.col("homeTeam")).select(
    "matchId", "competition", "season", "team",
    F.col("homeTeamGoals").alias("goalsScored"),  # Home team goals scored
    F.col("awayTeamGoals").alias("goalsConceded")  # Home team goals conceded
).unionByName(
    matchDF.withColumn("team", F.col("awayTeam")).select(
        "matchId", "competition", "season", "team",
        F.col("awayTeamGoals").alias("goalsScored"),  # Away team goals scored
        F.col("homeTeamGoals").alias("goalsConceded")  # Away team goals conceded
    )
)

# Step 2: Determine match results (win, draw, loss)
teamsDF = teamsDF.withColumn(
    "result",
    F.when(F.col("goalsScored") > F.col("goalsConceded"), "win")
    .when(F.col("goalsScored") == F.col("goalsConceded"), "draw")
    .otherwise("loss")
)

# Step 3: Assign points based on match results
teamsDF = teamsDF.withColumn(
    "points",
    F.when(F.col("result") == "win", 3)
    .when(F.col("result") == "draw", 1)
    .otherwise(0)
)

# Step 4: Aggregate results by season, league, and team
seasonDF = teamsDF.groupBy("competition", "season", "team").agg(
    F.count("*").alias("games").cast(IntegerType()),
    F.sum(F.when(F.col("result") == "win", 1).otherwise(0)).cast(IntegerType()).alias("wins"),
    F.sum(F.when(F.col("result") == "draw", 1).otherwise(0)).cast(IntegerType()).alias("draws"),
    F.sum(F.when(F.col("result") == "loss", 1).otherwise(0)).cast(IntegerType()).alias("losses"),
    F.sum("goalsScored").alias("goalsScored").cast(IntegerType()),
    F.sum("goalsConceded").alias("goalsConceded").cast(IntegerType()),
    F.sum("points").alias("points").cast(IntegerType())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 5 - English Premier League table
# MAGIC
# MAGIC Using the season data frame from basic task 4 calculate the final league table for `English Premier League` in season `2017-2018`.
# MAGIC
# MAGIC The result should be given as data frame which is ordered by the team's classification for the season.
# MAGIC
# MAGIC A team is classified higher than the other team if one of the following is true:
# MAGIC
# MAGIC - The team has a higher number of total points than the other team
# MAGIC - The team has an equal number of points, but have a better goal difference than the other team
# MAGIC - The team has an equal number of points and goal difference, but have more goals scored in total than the other team
# MAGIC
# MAGIC Goal difference is the difference between the number of goals scored for and against the team.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team.<br>
# MAGIC It should include the following columns (several columns renamed trying to match the [league table in Wikipedia](https://en.wikipedia.org/wiki/2017%E2%80%9318_Premier_League#League_table)):
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Pos         | integer     | The classification of the team |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC
# MAGIC The goal difference should be given as a string with an added `+` at the beginning if the difference is positive, similarly to the table in the linked Wikipedia article.

# COMMAND ----------

# Define a window specification for ranking based on the sorting criteria
rankWindow = Window.orderBy(
    F.col("points").desc(),
    (F.col("goalsScored") - F.col("goalsConceded")).desc(),
    F.col("goalsScored").desc()
)

# Filter data for the English Premier League first
englandDF = seasonDF.filter(F.col("competition") == "English Premier League")

englandDF = englandDF.withColumn(
    "Pos", F.row_number().over(rankWindow)
).withColumn(
    "GD", F.format_string("%+d", F.col("goalsScored") - F.col("goalsConceded"))
).selectExpr(
    "Pos",
    "team AS Team",
    "games AS Pld",
    "wins AS W",
    "draws AS D",
    "losses AS L",
    "goalsScored AS F",
    "goalsConceded AS A",
    "GD",
    "points AS Pts"
)

print("English Premier League table for season 2017-2018")
englandDF.show(20, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 6: Calculate the number of passes
# MAGIC
# MAGIC This task involves going back to the event data frame and counting the number of passes each team made in each match. A pass is considered successful if it is marked as `Accurate`.
# MAGIC
# MAGIC Using the event data frame from basic task 2, calculate the total number of passes as well as the total number of successful passes for each team in each match.<br>
# MAGIC The resulting data frame should contain one row for each team in each match, i.e., two rows for each match. It should include the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | matchId     | integer     | A unique id for the match |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | season      | string      | The season |
# MAGIC | team        | string      | The name of the team |
# MAGIC | totalPasses | integer     | The total number of passes the team attempted in the match |
# MAGIC | successfulPasses | integer | The total number of successful passes made by the team in the match |
# MAGIC
# MAGIC You can assume that each team had at least one pass attempt in each match they played.

# COMMAND ----------

passStatsDF = eventDF.filter(F.col("event") == "Pass").groupBy("matchId", "competition", "season", "eventTeam").agg(
    F.sum(F.expr("array_contains(tags, 'Accurate')::int")).alias("successfulPasses"),  # Count accurate passes
    F.count("*").alias("totalPasses")  # Count all passes
)

matchPassDF = passStatsDF.withColumnRenamed("eventTeam", "team")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 7: Teams with the worst passes
# MAGIC
# MAGIC Using the match pass data frame from basic task 6 find the teams with the lowest average ratio for successful passes over the season `2017-2018` for each league.
# MAGIC
# MAGIC The ratio for successful passes over a single match is the number of successful passes divided by the number of total passes.<br>
# MAGIC The average ratio over the season is the average of the single match ratios.
# MAGIC
# MAGIC Give the result as a data frame that has one row for each league-team pair with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | team        | string      | The name of the team |
# MAGIC | passSuccessRatio | double | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the lowest ratio for passes is given first.

# COMMAND ----------

lowestPassSuccessRatioDF = matchPassDF.groupBy("competition", "team").agg(
    F.round((F.sum("successfulPasses") / F.sum("totalPasses") * 100), 2).alias("passSuccessRatio")
).orderBy("passSuccessRatio")

print("The teams with the lowest ratios for successful passes for each league in season 2017-2018:")
lowestPassSuccessRatioDF.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 8: The best teams
# MAGIC
# MAGIC For this task the best teams are determined by having the highest point average per match.
# MAGIC
# MAGIC Using the data frames created in the previous tasks find the two best teams from each league in season `2017-2018` with their full statistics.
# MAGIC
# MAGIC Give the result as a data frame with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | League      | string      | The name of the league |
# MAGIC | Pos         | integer     | The classification of the team within their league |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC | Avg         | double      | The average points per match gained by the team |
# MAGIC | PassRatio   | double      | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the highest point average per match is given first.

# COMMAND ----------

# Step 1: Aggregate the statistics for each team in each league
teamStatsDF = seasonDF.groupBy("competition", "team").agg(
    F.sum("points").alias("Pts"),  # Total points
    F.sum("games").alias("Pld"),  # Total games played
    F.sum("wins").alias("W"),     # Total wins
    F.sum("draws").alias("D"),    # Total draws
    F.sum("losses").alias("L"),   # Total losses
    F.sum("goalsScored").alias("GF"),  # Total goals scored
    F.sum("goalsConceded").alias("GA"),  # Total goals conceded
    F.expr("SUM(goalsScored) - SUM(goalsConceded)").alias("GD")  # Goal difference
)

# Step 2: Calculate average points per match for each team
teamStatsDF = teamStatsDF.withColumn("Avg", F.col("Pts") / F.col("Pld"))

# Step 3: Calculate Pass Success Ratio for each team (using the previously computed matchPassDF)
passStatsDF = matchPassDF.groupBy("competition", "team").agg(
    F.sum("successfulPasses").alias("totalSuccessful"),
    F.sum("totalPasses").alias("totalAttempts")
).withColumn(
    "PassRatio", (F.col("totalSuccessful") / F.col("totalAttempts"))
)

# Step 4: Join the team statistics with pass success ratio
teamStatsDF = teamStatsDF.join(passStatsDF, on=["competition", "team"], how="left")

# Step 5: Rank teams by average points per match in each league
teamStatsDF = teamStatsDF.withColumn(
    "rank", F.rank().over(
        Window.partitionBy("competition").orderBy(F.col("Avg").desc())
    )
)

# Step 6: Filter top 2 teams per league
topTeamsDF = teamStatsDF.filter(F.col("rank") <= 2)

# Step 7: Select and reorder columns
bestDF = topTeamsDF.select(
    F.col("team").alias("Team"),
    F.col("competition").alias("League"),
    F.col("rank").alias("Pos"),
    F.col("Pld").cast(IntegerType()), 
    F.col("W").cast(IntegerType()),    
    F.col("D").cast(IntegerType()),  
    F.col("L").cast(IntegerType()),
    F.col("GF").cast(IntegerType()),  
    F.col("GA").cast(IntegerType()),  
    F.concat(
        F.when(F.col("GD") > 0, F.lit("+")).otherwise(F.lit("")),
        F.col("GD").cast("string")
    ).alias("GD"),
    F.col("Pts").cast(IntegerType()),  
    F.round(F.col("Avg"), 2).alias("Avg"),
    F.round(F.col("PassRatio") * 100, 2).alias("PassRatio") 
)

# Step 8: Order by average points per match
bestDF = bestDF.orderBy(F.col("Avg").desc())

print("The top 2 teams for each league in season 2017-2018")
bestDF.show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 2 - Further tasks with football data (2 points)
# MAGIC
# MAGIC This advanced task continues with football event data from the basic tasks. In addition, there are two further related datasets that are used in this task.
# MAGIC
# MAGIC A Parquet file at folder `assignment/football/matches.parquet` in the [Shared container] contains information about which players were involved on each match including information on the substitutions made during the match.
# MAGIC
# MAGIC Another Parquet file at folder `assignment/football/players.parquet` in the [Shared container] contains information about the player names, default roles when playing, and their birth areas.
# MAGIC
# MAGIC #### Columns in the additional data
# MAGIC
# MAGIC The match dataset (`assignment/football/matches.parquet`) has one row for each match and each row has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | matchId      | integer     | A unique id for the match |
# MAGIC | competition  | string      | The name of the league |
# MAGIC | season       | string      | The season the match was played |
# MAGIC | roundId      | integer     | A unique id for the round in the competition |
# MAGIC | gameWeek     | integer     | The gameWeek of the match |
# MAGIC | date         | date        | The date the match was played |
# MAGIC | status       | string      | The status of the match, `Played` if the match has been played |
# MAGIC | homeTeamData | struct      | The home team data, see the table below for the attributes in the struct |
# MAGIC | awayTeamData | struct      | The away team data, see the table below for the attributes in the struct |
# MAGIC | referees     | struct      | The referees for the match |
# MAGIC
# MAGIC Both team data columns have the following inner structure:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | team         | string      | The name of the team |
# MAGIC | coachId      | integer     | A unique id for the coach of the team |
# MAGIC | lineup       | array of integers | A list of the player ids who start the match on the field for the team |
# MAGIC | bench        | array of integers | A list of the player ids who start the match on the bench, i.e., the reserve players for the team |
# MAGIC | substitution1 | struct     | The first substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution2 | struct     | The second substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution3 | struct     | The third substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC
# MAGIC Each substitution structs have the following inner structure:
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerIn     | integer     | The id for the player who was substituted from the bench into the field, i.e., this player started playing after this substitution |
# MAGIC | playerOut    | integer     | The id for the player who was substituted from the field to the bench, i.e., this player stopped playing after this substitution |
# MAGIC | minute       | integer     | The minute from the start of the match the substitution was made.<br>Values of 45 or less indicate that the substitution was made in the first half of the match,<br>and values larger than 45 indicate that the substitution was made on the second half of the match. |
# MAGIC
# MAGIC The player dataset (`assignment/football/players.parquet`) has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerId     | integer     | A unique id for the player |
# MAGIC | firstName    | string      | The first name of the player |
# MAGIC | lastName     | string      | The last name of the player |
# MAGIC | birthArea    | string      | The birth area (nation or similar) of the player |
# MAGIC | role         | string      | The main role of the player, either `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
# MAGIC | foot         | string      | The stronger foot of the player |
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In a football match both teams have 11 players on the playing field or pitch at the start of the match. Each team also have some number of reserve players on the bench at the start of the match. The teams can make up to three substitution during the match where they switch one of the players on the field to a reserve player. (Currently, more substitutions are allowed, but at the time when the data is from, three substitutions were the maximum.) Any player starting the match as a reserve and who is not substituted to the field during the match does not play any minutes and are not considered involved in the match.
# MAGIC
# MAGIC For this task the length of each match should be estimated with the following procedure:
# MAGIC
# MAGIC - Only the additional time added to the second half of the match should be considered. I.e., the length of the first half is always considered to be 45 minutes.
# MAGIC - The length of the second half is to be considered as the last event of the half rounded upwards towards the nearest minute.
# MAGIC     - I.e., if the last event of the second half happens at 2845 seconds (=47.4 minutes) from the start of the half, the length of the half should be considered as 48 minutes. And thus, the full length of the entire match as 93 minutes.
# MAGIC
# MAGIC A personal plus-minus statistics for each player can be calculated using the following information:
# MAGIC
# MAGIC - If a goal was scored by the player's team when the player was on the field, `add 1`
# MAGIC - If a goal was scored by the opponent's team when the player was on the field, `subtract 1`
# MAGIC - If a goal was scored when the player was a reserve on the bench, `no change`
# MAGIC - For any event that is not a goal, or is in a match that the player was not involved in, `no change`
# MAGIC - Any substitutions is considered to be done at the start of the given minute.
# MAGIC     - I.e., if the player is substituted from the bench to the field at minute 80 (minute 35 on the second half), they were considered to be on the pitch from second 2100.0 on the 2nd half of the match.
# MAGIC - If a goal was scored in the additional time of the first half of the match, i.e., the goal event period is `1H` and event time is larger than 2700 seconds, some extra considerations should be taken into account:
# MAGIC     - If a player is substituted into the field at the beginning of the second half, `no change`
# MAGIC     - If a player is substituted off the field at the beginning of the second half, either `add 1` or `subtract 1` depending on team that scored the goal
# MAGIC     - Any player who is substituted into the field at minute 45 or later is only playing on the second half of the match.
# MAGIC     - Any player who is substituted off the field at minute 45 or later is considered to be playing the entire first half including the additional time.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to use the football event data and the additional datasets to determine the following:
# MAGIC
# MAGIC - The players with the most total minutes played in season 2017-2018 for each player role
# MAGIC     - I.e., the player in Goalkeeper role who has played the longest time across all included leagues. And the same for the other player roles (Defender, Midfielder, and Forward)
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `role`: the player role
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `minutes`: the total minutes the player played during season 2017-2018
# MAGIC - The players with higher than `+65` for the total plus-minus statistics in season 2017-2018
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `role`: the player role
# MAGIC         - `plusMinus`: the total plus-minus statistics for the player during season 2017-2018
# MAGIC
# MAGIC It is advisable to work towards the target results using several intermediate steps.

# COMMAND ----------

# mostMinutesDF: DataFrame = ???

# print("The players with the most minutes played in season 2017-2018 for each player role:")
# mostMinutesDF.show(truncate=False)

# COMMAND ----------

# topPlayers: DataFrame = ???

# print("The players with higher than +65 for the plus-minus statistics in season 2017-2018:")
# topPlayers.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Image data and pixel colors (2 points)
# MAGIC
# MAGIC This advanced task involves loading in PNG image data and complementing JSON metadata into Spark data structure. And then determining the colors of the pixels in the images, and finding the answers to several color related questions.
# MAGIC
# MAGIC The folder `assignment/openmoji/color` in the [Shared container] contains collection of PNG images from [OpenMoji](https://openmoji.org/) project.
# MAGIC
# MAGIC The JSON Lines formatted file `assignment/openmoji/openmoji.jsonl` contains metadata about the image collection. Only a portion of the images are included as source data for this task, so the metadata file contains also information about images not considered in this task.
# MAGIC
# MAGIC #### Data description and helper functions
# MAGIC
# MAGIC The image data considered in this task can be loaded into a Spark data frame using the `image` format: [https://spark.apache.org/docs/3.5.0/ml-datasource.html](https://spark.apache.org/docs/3.5.0/ml-datasource.html). The resulting data frame contains a single column which includes information about the filename, image size as well as the binary data representing the image itself. The Spark documentation page contains more detailed information about the structure of the column.
# MAGIC
# MAGIC Instead of using the images as source data for machine learning tasks, the binary image data is accessed directly in this task.<br>
# MAGIC You are given two helper functions to help in dealing with the binary data:
# MAGIC
# MAGIC - Function `toPixels` takes in the binary image data and the number channels used to represent each pixel.
# MAGIC     - In the case of the images used in this task, the number of channels match the number bytes used for each pixel.
# MAGIC     - As output the function returns an array of strings where each string is hexadecimal representation of a single pixel in the image.
# MAGIC - Function `toColorName` takes in a single pixel represented as hexadecimal string.
# MAGIC     - As output the function returns a string with the name of the basic color that most closely represents the pixel.
# MAGIC     - The function uses somewhat naive algorithm to determine the name of the color, and does not always give correct results.
# MAGIC     - Many of the pixels in this task have a lot of transparent pixels. Any such pixel is marked as the color `None` by the function.
# MAGIC
# MAGIC With the help of the given functions it is possible to transform the binary image data to an array of color names without using additional libraries or knowing much about image processing.
# MAGIC
# MAGIC The metadata file given in JSON Lines format can be loaded into a Spark data frame using the `json` format: [https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html](https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html). The attributes used in the JSON data are not described here, but are left for you to explore. The original regular JSON formatted file can be found at [https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json](https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json).
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to combine the image data with the JSON data, determine the image pixel colors, and the find the answers to the following questions:
# MAGIC
# MAGIC - Which four images have the most colored non-transparent pixels?
# MAGIC - Which five images have the lowest ratio of colored vs. transparent pixels?
# MAGIC - What are the three most common colors in the Finnish flag image (annotation: `flag: Finland`)?
# MAGIC     - And how many percentages of the colored pixels does each color have?
# MAGIC - How many images have their most common three colors as, `Blue`-`Yellow`-`Black`, in that order?
# MAGIC - Which five images have the most red pixels among the image group `activities`?
# MAGIC     - And how many red pixels do each of these images have?
# MAGIC
# MAGIC It might be advisable to test your work-in-progress code with a limited number of images before using the full image set.<br>
# MAGIC You are free to choose your own approach to the task: user defined functions with data frames, RDDs/Datasets, or combination of both.
# MAGIC
# MAGIC Note that the currently the Python helper functions do not exactly match the Scala versions, and thus the answers to the questions might not quite match the given example results in the example output notebook.

# COMMAND ----------

# separates binary image data to an array of hex strings that represent the pixels
# assumes 8-bit representation for each pixel (0x00 - 0xff)
# with `channels` attribute representing how many bytes is used for each pixel
def toPixels(data: bytes, channels: int) -> List[str]:
    return [
        "".join([
            f"{data[index+byte]:02X}"
            for byte in range(0, channels)
        ])
        for index in range(0, len(data), channels)
    ]

# COMMAND ----------

# naive implementation of picking the name of the pixel color based on the input hex representation of the pixel
# only works for OpenCV type CV_8U (mode=24) compatible input
def toColorName(hexString: str) -> str:
    # mapping of RGB values to basic color names
    colors: Dict[Tuple[int, int, int], str] = {
        (0, 0, 0):     "Black",  (0, 0, 128):     "Blue",   (0, 0, 255):     "Blue",
        (0, 128, 0):   "Green",  (0, 128, 128):   "Green",  (0, 128, 255):   "Blue",
        (0, 255, 0):   "Green",  (0, 255, 128):   "Green",  (0, 255, 255):   "Blue",
        (128, 0, 0):   "Red",    (128, 0, 128):   "Purple", (128, 0, 255):   "Purple",
        (128, 128, 0): "Green",  (128, 128, 128): "Gray",   (128, 128, 255): "Purple",
        (128, 255, 0): "Green",  (128, 255, 128): "Green",  (128, 255, 255): "Blue",
        (255, 0, 0):   "Red",    (255, 0, 128):   "Pink",   (255, 0, 255):   "Purple",
        (255, 128, 0): "Orange", (255, 128, 128): "Orange", (255, 128, 255): "Pink",
        (255, 255, 0): "Yellow", (255, 255, 128): "Yellow", (255, 255, 255): "White"
    }

    # helper function to round values of 0-255 to the nearest of 0, 128, or 255
    def roundColorValue(value: int) -> int:
        if value < 85:
            return 0
        if value < 170:
            return 128
        return 255

    validString: bool = re.match(r"[0-9a-fA-F]{8}", hexString) is not None
    if validString:
        # for OpenCV type CV_8U (mode=24) the expected order of bytes is BGRA
        blue: int = roundColorValue(int(hexString[0:2], 16))
        green: int = roundColorValue(int(hexString[2:4], 16))
        red: int = roundColorValue(int(hexString[4:6], 16))
        alpha: int = int(hexString[6:8], 16)

        if alpha < 128:
            return "None"  # any pixel with less than 50% opacity is considered as color "None"
        return colors[(red, green, blue)]

    return "None"  # any input that is not in valid format is considered as color "None"

# COMMAND ----------

# Paths to the image and metadata
imgPath = ".../assignment/openmoji/color"
metadataPath = ".../assignment/openmoji/metadata/openmoji.jsonl"

# Load image and metadata
imageDF = spark.read.format("image").option("dropInvalid", True).load(imgPath)
metadataDF = spark.read.json(metadataPath)

# Extract the hexcode from the image path (after the last '/' and before the extension)
imageDF = imageDF.withColumn(
    "hexcode",
    F.substring_index(F.substring_index(F.col("image.origin"), "/", -1), ".", 1)
)

# Rename 'hexcode' in metadataDF to avoid ambiguity
metadataDF = metadataDF.withColumnRenamed("hexcode", "metadata_hexcode")

# Join imageDF with metadataDF on the 'hexcode' column from imageDF and 'metadata_hexcode' from metadataDF
imageDFWithAnnotations = imageDF.join(metadataDF, imageDF["hexcode"] == metadataDF["metadata_hexcode"], "inner")

# Define UDFs for pixel extraction and color conversion
toPixelsUdf = F.udf(lambda data: toPixels(data, 4), ArrayType(StringType()))
toColorNameUdf = F.udf(lambda hexList: [toColorName(pixel) for pixel in hexList], ArrayType(StringType()))

# Extract pixel data and map to color names
imageWithColorsDf = imageDFWithAnnotations.withColumn(
    "pixels", toPixelsUdf(F.col("image.data"))
).withColumn(
    "colors", toColorNameUdf(F.col("pixels"))
).withColumn(
    "nonTransparentCount",
    F.expr("size(colors) - size(filter(colors, c -> c = 'None'))")
).withColumn(
    "colorRatio",
    F.expr("nonTransparentCount / size(colors)")
)

# Get the top 4 images with the most non-transparent pixels
mostColoredPixelsDf = imageWithColorsDf.orderBy(F.col("nonTransparentCount").desc()).limit(4)

# Collect annotations for the top 4 images
mostColoredPixels = [row["annotation"] for row in mostColoredPixelsDf.select("annotation").collect()]

print("The annotations for the four images with the most colored non-transparent pixels:")
for annotation in mostColoredPixels:
    print(f"- {annotation}")
print("============================================================")

# Bottom 5 images by color ratio (colored vs transparent pixels)
imageWithColorsDf = imageWithColorsDf.withColumn(
    "colorRatio",
    F.expr("nonTransparentCount / size(colors)")
)

# Get the bottom 5 images with lowest ratio of colored to transparent pixels
leastColoredPixelsDf = imageWithColorsDf.orderBy(F.col("colorRatio").asc()).limit(5)

# Collect annotations for the bottom 5
leastColoredPixels = [row["annotation"] for row in leastColoredPixelsDf.select("annotation").collect()]

print("The annotations for the five images having the lowest ratio of colored vs. transparent pixels:")
for annotation in leastColoredPixels:
    print(f"- {annotation}")


# COMMAND ----------

# The three most common colors in the Finnish flag image:
finnishFlagColors: List[str] = imageWithColorsDf.filter(
    F.col("annotation") == "flag: Finland"
).select("colors").collect()[0]["colors"]

# Filter out 'None' (transparent pixels)
finnishFlagColors = [color for color in finnishFlagColors if color != 'None']

# The percentages of the colored pixels for each common color in the Finnish flag image:
colorCounts = Counter(finnishFlagColors )

# Get the three most common colors
mostCommonColors = colorCounts.most_common(3)

# Calculate the percentages of the colored pixels for each common color
totalColors = len(finnishFlagColors )
finnishColorShares = [(color, count / totalColors) for color, count in mostCommonColors]

print("The colors and their percentage shares in the image for the Finnish flag:")
for colors in zip(finnishFlagColors, finnishColorShares):
    color = colors[0]
    share = colors[1]
    print(f"- color: {color}, share: {share}")
print("============================================================")

# Count the number of images with their most common three colors as Blue-Yellow-Black in that exact order
# First, filter out images where Blue, Yellow, and Black are the three most common colors in that order
def get_most_common_colors(colors):
    # Count the occurrences of each color
    colorCounts = Counter(color for color in colors if color != 'None')
    # Get the three most common colors, sorted by frequency
    commonColors = [color for color, _ in colorCounts.most_common(3)]
    # Check if the three most common colors are Blue, Yellow, Black in this exact order
    return commonColors[:3] == ['Blue', 'Yellow', 'Black']

# Register the UDF to compare the most common colors
most_common_colors_udf = F.udf(get_most_common_colors, BooleanType())

# Apply the UDF to filter images where the three most common colors are Blue, Yellow, Black in that order
blueYellowBlackCount = imageWithColorsDf.filter(
    most_common_colors_udf(F.col('colors'))
).count()

print(f"The number of images that have, Blue-Yellow-Black, as the most common colors: {blueYellowBlackCount}")

# COMMAND ----------

# Add a new column with the count of red pixels for each image directly using Spark functions
imageWithColorsDf = imageWithColorsDf.withColumn(
    "red_pixel_count", 
    F.size(F.filter(F.col("colors"), lambda x: x == "Red"))
)

# The annotations for the five images with the most red pixels among the image group activities:
redImageNames = (
    imageWithColorsDf.filter(F.col("group") == "activities")
    .filter(F.array_contains(F.col("colors"), "Red"))
    .groupBy("annotation")
    .agg(F.sum("red_pixel_count").alias("total_red_pixel_count"))
    .orderBy(F.col("total_red_pixel_count").desc())
    .limit(5)
    .select("annotation", "total_red_pixel_count") 
    .collect()
)

print("The annotations and red pixel counts for the five images with the most red pixels among the image group 'activities':")
for row in redImageNames:
    annotation = row["annotation"]
    red_pixel_count = row["total_red_pixel_count"]
    print(f"- {annotation} (red pixels: {red_pixel_count})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 4 - Machine learning tasks (2 points)
# MAGIC
# MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the [ProCem](https://www.senecc.fi/projects/procem-2) research project is used as the training and test data. Similar data in a slightly different format was used in the first tasks of weekly exercise 3.
# MAGIC
# MAGIC The folder `assignment/energy/procem_13m.parquet` in the [Shared container] contains the time series data in Parquet format.
# MAGIC
# MAGIC #### Data description
# MAGIC
# MAGIC The dataset contains time series data from a period of 13 months (from the beginning of May 2023 to the end of May 2024). Each row contains the average of the measured values for a single minute. The following columns are included in the data:
# MAGIC
# MAGIC | column name        | column type   | description |
# MAGIC | ------------------ | ------------- | ----------- |
# MAGIC | time               | long          | The UNIX timestamp in second precision |
# MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
# MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
# MAGIC | wind_speed         | double        | The wind speed measured by the weather station on top of Sähkötalo (`m/s`) |
# MAGIC | power_tenants      | double        | The total combined electricity power used by the tenants on Kampusareena (`W`) |
# MAGIC | power_maintenance  | double        | The total combined electricity power used by the building maintenance systems on Kampusareena (`W`) |
# MAGIC | power_solar_panels | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
# MAGIC | electricity_price  | double        | The market price for electricity in Finland (`€/MWh`) |
# MAGIC
# MAGIC There are some missing values that need to be removed before using the data for training or testing. However, only the minimal amount of rows should be removed for each test case.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC - The main task is to train and test a machine learning model with [Random forest classifier](https://spark.apache.org/docs/3.5.0/ml-classification-regression.html#random-forests) in six different cases:
# MAGIC     - Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input
# MAGIC     - Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
# MAGIC - For each of the six case you are asked to:
# MAGIC     1. Clean the source dataset from rows with missing values.
# MAGIC     2. Split the dataset into training and test parts.
# MAGIC     3. Train the ML model using a Random forest classifier with case-specific input and prediction.
# MAGIC     4. Evaluate the accuracy of the model with Spark built-in multiclass classification evaluator.
# MAGIC     5. Further evaluate the accuracy of the model with a custom build evaluator which should do the following:
# MAGIC         - calculate the percentage of correct predictions
# MAGIC             - this should correspond to the accuracy value from the built-in accuracy evaluator
# MAGIC         - calculate the percentage of predictions that were at most one away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be `4`, `5`, or `6`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be `12`, `1`, or `2`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be `11`, `12`, or `1`
# MAGIC         - calculate the percentage of predictions that were at most two away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be from `3` to `7`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be from `11` to `12` and from `1` to `3`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be from `10` to `12` and from `1` to `2`
# MAGIC         - calculate the average probability the model predicts for the correct value
# MAGIC             - the probabilities for a single prediction can be found from the `probability` column after the predictions have been made with the model
# MAGIC - As the final part of this advanced task, you are asked to do the same experiments (training+evaluation) with two further cases of your own choosing:
# MAGIC     - you can decide on the input columns yourself
# MAGIC     - you can decide the predicted attribute yourself
# MAGIC     - you can try some other classifier other than the random forest one if you want
# MAGIC
# MAGIC In all cases you are free to choose the training parameters as you wish.<br>
# MAGIC Note that it is advisable that while you are building your task code to only use a portion of the full 13-month dataset in the initial experiments.

# COMMAND ----------

path: str = ".../assignment/energy/procem_13m.parquet"
weatherDF: DataFrame = spark.read.parquet(path)

# Clean the source dataset from rows with missing values.
weatherDF = weatherDF.dropna()

# Extract additional time-based features: month, hour, and weekday from the timestamp.
weatherDF = weatherDF.withColumn("month", F.from_unixtime(F.col("time"), "MM").cast("int")) \
    .withColumn("hour", F.from_unixtime(F.col("time"), "HH").cast("int")) \
    .withColumn("weekday", F.from_unixtime(F.col("time"), "dd").cast("int")) \
    .withColumn("peak-hour",F.when((F.col("hour") >= 8) & (F.col("hour") <= 18), 1) \
    .otherwise(0)
)

# Define input features for the ML model:
# input1: Weather-related features, input2: Power-related features,
# input3: Combined weather, power, and price, input4: Weather and price features for peak-hour prediction.
input1 = "temperature, humidity, wind_speed"
input2 = "power_tenants, power_maintenance, power_solar_panels"
input3 = input1 + ", " + input2 + ", " + "electricity_price"
input4 = "temperature, humidity, wind_speed, electricity_price"

# Function to train a Random Forest model to predict the given target variable using the provided input features.
def trainMLModel(input, time):
    print(f"Training the ML model to predict '{time}' using the input: '{input}'")

    # Use VectorAssembler to combine input features into a single vector column required for model training.
    assembler = VectorAssembler(inputCols=input.split(", "), outputCol="features")
    transformedDF = assembler.transform(weatherDF)

    # Split the dataset into training and test parts
    trainDF, testDF = transformedDF.randomSplit([0.8, 0.2], seed=1)

    # Train the Random Forest classifier
    rf = RandomForestClassifier(labelCol=time, featuresCol="features", numTrees=100)
    model = rf.fit(trainDF)

    # Make predictions
    predictions = model.transform(testDF)

    # Evaluate the model performance using F1-score for 'peak-hour' and accuracy for other target variables.
    if(time == "peak-hour"):
        f1 = MulticlassClassificationEvaluator(labelCol=time, predictionCol="prediction", metricName="f1").evaluate(predictions)
        print(f"Test set accuracy = {f1}")
    else:
        accuracy = MulticlassClassificationEvaluator(labelCol=time, predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        print(f"Test set accuracy = {accuracy}")

    # Evaluator that calculates the percentage of correct predictions.
    def correctPredEvaluator(predictions):
        correct = predictions.filter(F.col(time) == F.col("prediction")).count()
        total = predictions.count()
        percentage = correct / total * 100
        return round(percentage, 2)

    # Evaluator that checks if the prediction is within one unit of the true value.
    def oneAwayEvaluator(predictions):
        if(time == "peak-hour"):
            return "N/A"
        correct = predictions.filter(F.expr(f"abs({time} - prediction) <= 1 OR abs({time} - prediction) == 11")).count()
        total = predictions.count()
        percentage = correct / total * 100
        return round(percentage, 2)

    # Evaluator that checks if the prediction is within two units of the true value.
    def twoAwayEvaluator(predictions):
        if(time == "peak-hour"):
            return "N/A"
        correct = predictions.filter(F.expr(f"abs({time} - prediction) <= 2 OR abs({time} - prediction) >= 10")).count()
        total = predictions.count()
        percentage = correct / total * 100
        return round(percentage, 2)
    
    # Evaluator that calculates the average of the correct class probabilities for predictions.
    def averageProbabilityEvaluator(predictions):
        extract_prob = F.udf(lambda prob, time: float(prob[int(time)]), DoubleType())
        predictions = predictions.withColumn("correct_probability", extract_prob(F.col("probability"), F.col(time)))

        average = predictions.agg(F.avg("correct_probability").alias("avg_correct_probability")).collect()[0]["avg_correct_probability"]
        return round(average, 4)
    
    # Evaluator that calculates the Area Under the ROC Curve (AUC-ROC) for the model, used for binary classification.
    def aucRocEvaluator(predictions):
        if(time != "peak-hour"):
            return "N/A"
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol=time, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        auc = binary_evaluator.evaluate(predictions)
        return round(auc, 4)

    def customEvaluator(predictions):
        accuracy = correctPredEvaluator(predictions)
        oneAway = oneAwayEvaluator(predictions)
        twoAway = twoAwayEvaluator(predictions)
        average = averageProbabilityEvaluator(predictions)
        auc_roc = aucRocEvaluator(predictions)

        return accuracy, oneAway, twoAway, average, auc_roc

    correct, within_one, within_two, avg_prob, auc_roc = customEvaluator(predictions)

    classifier = "Random Forest"
    # Create a DataFrame to store evaluation results
    resultDF: DataFrame = spark.createDataFrame([(classifier, input, time, correct, within_one, within_two, auc_roc, avg_prob)], \
            ["Classifier", "Input", "Label Column", "Correct", "Within One", "Within Two", "AUC-ROC", "Average Probability"])
    
    return resultDF

resultDF1m = trainMLModel(input1, "month")
resultDF1d = trainMLModel(input1, "hour")
resultDF2m = trainMLModel(input2, "month")
resultDF2d = trainMLModel(input2, "hour")
resultDF3m = trainMLModel(input3, "month")
resultDF3d = trainMLModel(input3, "hour")
resultDF3w = trainMLModel(input3, "weekday")
resultDFPeak = trainMLModel(input4, "peak-hour")

# Combine results from different models and sort by accuracy.
finalResultDF = (
    resultDF1m
    .union(resultDF2m)
    .union(resultDF3m)
    .union(resultDF1d)
    .union(resultDF2d)
    .union(resultDF3d)
    .union(resultDF3w)
    .union(resultDFPeak)
    .sort(F.desc(F.col("Correct")))
)

finalResultDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The "peak-hour" prediction refers to predicting whether a given hour falls within the peak electricity usage hours. 
# MAGIC In this case, peak hours are defined as those between 8 AM and 6 PM, where electricity consumption is typically higher due to 
# MAGIC increased activity in homes and businesses. The model will predict a binary outcome:
# MAGIC  - 1: The hour is within the peak electricity usage window (8 AM to 6 PM).
# MAGIC  - 0: The hour is outside of the peak usage window (before 8 AM or after 6 PM).
# MAGIC
# MAGIC This prediction is useful for forecasting energy demand and optimizing electricity distribution, 
# MAGIC particularly in energy systems that experience fluctuating demand during different times of the day. (This explanation was created with ChatGPT)