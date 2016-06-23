package smctypes

// Fold determines the partitions of the samples.
type Fold struct {
	// Training, Alpha, and Correct are the list of samples used in each of the
	// parts of the StackMC procedure. The outer length is the data used per fold.
	// Inner length is the sample number to be used in that stage. Index refers
	// to a sample row.
	Train  []int // rows in Locations used for fitting
	Assess []int // rows in locations used for setting alpha
	Update []int // rows in locations used to update the data fit
}
