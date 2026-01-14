import pandas as pd
import numpy as np
import Levenshtein
import heapq
pd.options.mode.copy_on_write = True
class RankDesign():
    def __init__(self):
        self.metric_threshold_dict = {'interface_sc' : 0.55,
                                      'neg_interface_holo_apo_rmsd' : -1, # Negative because flipping all columns so higher is better
                                      'iptm' : 0.8,
                                      'ptm_apo' : 0.8,
                                      'plddt': 0.9,
                                      'iplddt': 0.9,
                                      'desired_epitope_coverage' : 0.8, # Boltz2 Recall 
                                      'true_desired_epitope_coverage' : 0.25, # Boltz2 Precision
                                      'desired_epitope_coverage_chai' : 0.4} # Chai F1 score (Future needs to indicate this in metric)
        # Dictionary of metrics and respective weights in terms of design criteria importance (smaller value means more important
        self.metrics_weights = {
                                # Tier 1: Primary Success Metrics (Weight = 1)
                                "neg_interface_dG": 1,              # Physics is king
                                "iptm": 1,                          # Model confidence in the interface
                                "desired_epitope_coverage": 1,      # You want to prioritize hitting the target
                                "desired_epitope_coverage_chai": 1, # In Blind Validation, should still hit the target
                                # Tier 2: Structural Quality (Weight = 2)
                                # These effectively count for "half" as much as Tier 1
                                "interface_sc": 2,                  # Shape complementarity
                                "neg_interface_dG_SASA_ratio": 2,  # Efficient binding
                                "neg_interface_holo_apo_rmsd": 2,   # Smaller absolute value means less induced fit
                                "ptm_apo": 2,                       # Model confidence in the apo structure
                                "true_desired_epitope_coverage": 2, # Want to prioritize Recall (Hitting all the desired epitope residues, with decent precision so minimal off-target contacts)
                                # Tier 3: Nice-to-Haves (Weight = 3 or 4)
                                "iplddt": 3,                        # Model confidence in the interface
                                "binding_interface_hbonds": 4,      # Counts are noisy; high count doesn't always mean better
                                "interface_packstat": 4,
                                }
        # Columns to flip signs for due to want all metrics to be formatted where higher value = better design
        self.cols_to_flip = ['interface_dG', 'interface_dG_SASA_ratio', 'interface_holo_apo_rmsd']
    
    def update_threshold(self, metric:str, threshold:float):
        """ Update threshold for a given metric """
        self.metric_threshold_dict[metric] = threshold
        
    def update_weights(self, metric:str, weight:int):
        """ Update weight for a given metric """
        self.metrics_weights[metric] = weight
    
    def create_designed_seq_col(self, df_designs:pd.DataFrame, design_mask: np.ndarray, seq_col:str = 'sequence'):
        """ 
            Create a column in df_designs with the "sequence_designed" as the column name
            "sequence_designed" refers to the sequence designed by the model. 
                Subset of the full sequence as the model does not touch specific user-defined regions (i.e CDRs for antibodies)
            Column is used to define the actual sequence as input for seq similarity calculations between designs
        """
        df_designs_copy = df_designs.copy()
        df_designs_copy['sequence_designed'] = df_designs_copy[seq_col].apply(lambda x: "".join([aa for i, aa in enumerate(x) 
                                                                                                 if design_mask[i]]))
        return df_designs_copy


    def flip_cols(self, df):
        """ Create negative versions of metrics such that all metrics are higher value is better """
        df_copy = df.copy()
        for col in self.cols_to_flip:
            df_copy[f"neg_{col}"] = df_copy[col] * -1
        return df_copy
    
    
    def pass_fail_design_metric(self, df_designs:pd.DataFrame, seq_col: str = 'sequence') -> pd.DataFrame:
        """
        Goal:
            Create a Boolean DataFrame with 
                Columns are the metrics in metric_threshold_dict
                Values are True if the design passes (>=) threshold and False if the design fails (<) threshold
        Args:
            df_designs (pd.DataFrame): DataFrame of designs as rows and some of the columns are the metrics in metric_threshold_dict. 
                                       Also, has the design_id_col as a column.
            design_id_col (str): Name of the column in df_designs that is the design ID. Default is 'sequence'.
        Returns:
            pd.DataFrame: Boolean DataFrame with the index being the design_id_col in df_designs
                          Columns are the metrics in metric_threshold_dict
                          Values are True if the design passes (>=) threshold and False if the design fails (<) threshold
                          Final Column is 'num_filters_passed' which is the number of filters passed for each design
        """
        # Check if design_id_col is in df_designs
        assert seq_col in df_designs.columns, f"Sequence column {seq_col} not found in df_designs"
    
        # Check if critical metrics are in df_designs
        critical_metrics = set(self.metric_threshold_dict.keys()) - {'iplddt', 'true_desired_epitope_coverage', 'desired_epitope_coverage_chai'}
        missing_metrics = critical_metrics.difference(df_designs.columns) # Difference is directional
        assert critical_metrics.issubset(df_designs.columns), f"Metrics: {missing_metrics} not found in df_designs"

        # Initialize the boolean DataFrame with the same index as the input DataFrame. Prevents misalignment when checking Threshold Pass/Fail
        df_bool = pd.DataFrame(index=df_designs.index)
    
        # Iterate through the metrics and thresholds
        for metric, threshold in self.metric_threshold_dict.items():
            if metric in df_designs.columns:
                # Create a Boolean column for the metric
                df_bool[metric] = df_designs[metric] >= threshold
            else:
                print(f"Warning: Metric {metric} not used as a filter. Skipping...")
        
        # Add a column for the number of filters passed
        df_bool['num_filters_passed'] = df_bool.sum(axis=1)
        df_bool[seq_col] = df_designs[seq_col]

        # Add num_filters_passed column to df_designs
        df_designs['num_filters_passed'] = df_bool['num_filters_passed']
        
        return df_bool, df_designs
    
    def rank_designs(self, df_designs: pd.DataFrame) -> pd.DataFrame:
        """
        Implements 'Worst-Case Ranking' Accounting for Number of Passed Filters for each Design.
    
            1. Designs are ranked first by how many filters they passed.
            2. Then they are ranked by the metric value.
            3. The rank is scaled by the metric's inverse importance (weight).
            4. The design's final score is its WORST (max) scaled rank.

        Args:
            df_designs (pd.DataFrame): DataFrame of designs as rows and some of the columns are the metrics in metric_threshold_dict.
        Returns:
            pd.DataFrame: df_designs with a new column 'max_weighted_rank' containing the worst-case rank for each design.
        """
        # Create a copy to avoid SettingWithCopy warnings on the original df
        df = df_designs.copy()
    
        # We will store the calculated ranks in a separate dataframe temporarily
        df_rank = pd.DataFrame(index=df.index)

        print(f"Ranking {len(df)} designs...")

        for metric, inverse_weight in self.metrics_weights.items():
            if metric not in df.columns:
                print(f"Warning: Metric '{metric}' not found in DataFrame. Skipping.")
                continue

            # --- THE CRITICAL 'OPTION B' STEP ---
            # We rank based on a Tuple: (num_filters_passed, metric_value)
            # Python compares tuples element-by-element. Example: (5, 0.9) is greater than (4, 0.99).
            # This forces designs with more passed filters to the top.
            # ascending=False: Higher Filter Count is better, Higher Metric is better.
            # method='min': Ties get the best possible rank (e.g. both are Rank 1).
        
            raw_rank = (df[['num_filters_passed', metric]]
                        .apply(tuple, axis=1)
                        .rank(method="min", ascending=False)
                        )
        
            # Apply Inverse Weighting
            # Metric Weight 1 (Critical) -> Rank / 1 -> Rank stays large (Bad)
            # Metric Weight 5 (Minor)    -> Rank / 5 -> Rank shrinks (Good)
            df_rank[f"rank_{metric}"] = raw_rank / inverse_weight

        # --- THE "WORST CASE" AGGREGATION ---
        # Find the maximum (worst) rank across all metrics for each design.
        # This identifies the design's "weakest link".
        df["max_weighted_rank"] = df_rank.max(axis=1)

        # --- FINAL SORT ---
        # 1. Primary: max_weighted_rank (Ascending -> Lower rank is better)
        # 2. Tie-Breaker: iptm (Descending -> Higher score is better)
        if "iptm" in df.columns:
            df_sorted = df.sort_values(
                by=["max_weighted_rank", "iptm"], 
                ascending=[True, False]
            )
        else:
            df_sorted = df.sort_values(by="max_weighted_rank", ascending=True)
    
        return df_sorted
    #------- Functions for Design Selection ------------------------------------------------
    def calculate_design_score(self, quality: float, diversity: float, alpha: float = 0.1):
        """ Calculates inverted score for design based on quality and diversity of design
            Formula: score = (1-alpha)*quality + alpha*diversity 
            Args:
                quality (float): Quality of design
                diversity (float): Diversity of design
                alpha (float): Weighting of Diversity vs Quality. Must be between 0 and 1. Default is 0.1.
            Returns:
                float: Score for design
        """
        assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"
        return (1-alpha)*quality + alpha*diversity

    def calculate_diversity_score(self, seq_a: str, seq_b: str):
        """ Calculates diversity score between two sequences
        Diversity score is calculated as the fraction of positions in the two sequences that are different.
        Args:
            seq_a (str): Sequence A
            seq_b (str): Sequence B
        Returns:
            float: Diversity score between two sequences
        """
        assert len(seq_a) == len(seq_b), "Sequences must be of equal length"
        lev_distance = Levenshtein.distance(seq_a, seq_b)
        diversity_score = lev_distance / len(seq_a)
        return diversity_score

    def get_top_n_diverse_designs(self, df_design_ranked: pd.DataFrame, top_n: int, alpha: float = 0.1) -> pd.DataFrame:
        """ Returns the top_n designs from df_design_ranked, while accounting for diversity of the designs
            Each Design evaluated based on Score = (1-alpha)*Quality + alpha*Diversity
            Ranges of Vals:
                Quality: 0 to 1
                Diversity: 0 to 1
                alpha: 0 to 1
                Score: 0 to 1
            Goal is to maximize this total score for all designs in the final top_n designs selected

            Args:
                df_design_ranked (pd.DataFrame): DataFrame of designs as rows and columns are metrics for designs or design information (seq, pdb_filename, etc.) 
                    df_design_ranked is already sorted based on max_weighted_rank and iptm. Sort = [max_weighted_rank : ascending, iptm: descending] so higher quality designs have smaller rank and larger iptm.
                    Must Include Columns:
                        'max_weighted_rank': measure of how well design meets all filters and reflects design's lowest performing   metric
                        'sequence_designed': portion of full sequence that was designable
                top_n (int): Number of top designs to return.
                alpha (float): Weighting of Diversity vs Quality. Must be between 0 and 1. Default is 0.1.
            Returns:
                pd.DataFrame: DataFrame of top_n designs with columns as metrics for designs or design information (seq, pdb_filename, etc.)
        """
        assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"
        assert top_n > 0, "top_n must be greater than 0"
        assert 'max_weighted_rank' in df_design_ranked.columns, "df_design_ranked must include column 'max_weighted_rank'"
        assert 'sequence_designed' in df_design_ranked.columns, "df_design_ranked must include column 'sequence_designed'"
    
        df_design_ranked_copy = df_design_ranked.reset_index(drop = True)

        # 1. Normalize Design Quality Ranks to be between 0 and 1 where higher quality designs closer to 1
        rank_min = df_design_ranked_copy['max_weighted_rank'].min()
        rank_max = df_design_ranked_copy['max_weighted_rank'].max()
        df_design_ranked_copy['quality_score'] = (1 - ((df_design_ranked_copy['max_weighted_rank'] - rank_min) 
                                                   / (rank_max - rank_min)))
        # 2. Take first row as the first design in the top_n designs
        selected_indices = []
        selected_indices.append(0) # Add first row since it is the top design that should be validated in lab
        seq_designed_regions = df_design_ranked_copy['sequence_designed'].to_list() # Save each design's designed regions
        # Move forward with remaining designs
        df_design_to_judge = df_design_ranked_copy.iloc[1:]

        # 3. Calculate Ideal Scores for each design based on Quality and Assuming Perfect Diversity (diversity = 1)
        # Since using a heap, need to invert the actual scores so that the lowest ideal score is the highest actual score
        # Heaps are restricted to only be min heaps
        df_design_to_judge['ideal_score'] = df_design_to_judge.apply(lambda x: self.calculate_design_score(quality=x['quality_score'],diversity=1, alpha = alpha), axis=1)
        df_design_to_judge['ideal_score'] = -df_design_to_judge['ideal_score']
        
        # 4. Create a heap of the ideal scores
        # Create list of tuples of (ideal_score, design_index) & store designed_seq regions
        ideal_score_ids = list(zip(df_design_to_judge['ideal_score'], df_design_to_judge.index))
        heapq.heapify(ideal_score_ids)

        # 5. Iterate through heap till user-specified number of designs is met
        design_scores = [1] # Store design scores and start with 1 as placeholder for first design
        num_passed_designs = 1
        while len(selected_indices) <=  (top_n - 1):
            # 1. Pop the lowest ideal score from the heap (Current best design)
            score, score_id_best_design = heapq.heappop(ideal_score_ids)
            # 2. Get the sequence designed region for the current best design
            seq_designed_curr_best = seq_designed_regions[score_id_best_design]
            
            # 3. Compute minimum diversity score of current best design to all other designs in selected indices
            # Want minimum because min diversity = max similarity = bleakest perspective of diversity score
            min_diversity_score = 1
            for selected_id in selected_indices:
                seq_designed_other = seq_designed_regions[selected_id]
                diversity_score = self.calculate_diversity_score(seq_designed_curr_best, seq_designed_other)
                if diversity_score < min_diversity_score:
                    min_diversity_score = diversity_score
                # Break if min_diversity = 0
                if min_diversity_score == 0:
                    break
            
            # 4. Calculate actual score for current best design
            quality_score_best_design = df_design_to_judge.loc[score_id_best_design]['quality_score']
            design_score_actual = self.calculate_design_score(quality = quality_score_best_design,
                                              diversity = min_diversity_score,
                                              alpha = alpha)
            neg_actual_score = -1 * design_score_actual # Must calculate negative actual score since heap is a min heap
            
            # 5. Compare neg actual score of current best design to ideal score of next current best design
            score_next_curr_best, score_id_next_curr_best =  ideal_score_ids[0]
            if neg_actual_score <= score_next_curr_best: # If current best design is better than next best design
                selected_indices.append(score_id_best_design) # Add current best design index to selected indices
                design_scores.append(design_score_actual)
                num_passed_designs += 1
                print(f'Added design {score_id_best_design} to selected indices. Counter: {num_passed_designs}')
            else: # Push the design's actual score along with its id back onto the heap
                heapq.heappush(ideal_score_ids, (neg_actual_score, score_id_best_design))
        
        df_top_designs = df_design_ranked_copy.loc[selected_indices]
        df_top_designs['design_score'] = design_scores
        return df_top_designs
    
    def run_filter_rank_pipeline(self, df_designs: pd.DataFrame, design_mask: np.ndarray, top_n: int, alpha: float = 0.1,
                                 seq_col: str = 'sequence'):
        """ Run the entire Protein Design Filtering & Ranking Pipeline 
            Args:
                df_designs (Pandas DataFrame): A dataframe containing the designs to be filtered and ranked
                top_n (int): The number of top designs to return
                alpha (float): The weight of the diversity score in the design score 
                               (Higher alpha = greater diversity, but lower quality)
            Returns:
                Pandas DataFrame: A dataframe containing the top_n designs filtered and ranked
        """
        df_designs_copy = df_designs.copy()
        # 1. Flip sign of lower is better metrics and add sequence_designed column
        df_designs_copy = self.flip_cols(df = df_designs_copy)
        df_designs_copy = self.create_designed_seq_col(df_designs = df_designs_copy, design_mask= design_mask, seq_col= seq_col)
        # 2. Create Boolean DataFrame of columns = metrics and values True/False for each design passing metric threshold
        df_bool, df_designs_copy = self.pass_fail_design_metric(df_designs = df_designs_copy, seq_col= seq_col)
        # 3. Rank designs based on Boolean DataFrame
        df_ranked_designs = self.rank_designs(df_designs = df_designs_copy)
        # 4. Extract top_n designs based on ranks and design similarity
        df_top_n_designs = self.get_top_n_diverse_designs(df_design_ranked = df_ranked_designs, top_n = top_n, alpha = alpha)
        return df_top_n_designs






    




