import pytest

from constraints import (
    ValidAssignmentConstraint,
    AllDifferentPieceConstraint,
    OnlyLastOrdersCanBeUnusedConstraint,
    NoHolesOfXConstraint,
)


@pytest.mark.parametrize("assignments", [[(0, 6, 3, 0)], [(0, 6, 3, 0), (1, 1, 1, 1)]])
def test_valid_assignments(assignments):
    assert AllDifferentPieceConstraint()(*assignments)
    assert OnlyLastOrdersCanBeUnusedConstraint()(*assignments)

    assert OnlyLastOrdersCanBeUnusedConstraint()(*assignments)
    assert ValidAssignmentConstraint(5, 12)(*assignments)
    assert NoHolesOfXConstraint(5, 12, 3)(*assignments)
