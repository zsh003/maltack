"""empty message

Revision ID: 0d8476f24939
Revises: 
Create Date: 2025-03-20 17:14:24.262818

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0d8476f24939'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('upload_history',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('file_name', sa.String(length=255), nullable=False),
    sa.Column('file_type', sa.String(length=50), nullable=False),
    sa.Column('environment', sa.String(length=255), nullable=False),
    sa.Column('upload_time', sa.DateTime(), nullable=False),
    sa.Column('threat_level', sa.String(length=20), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('upload_history')
    # ### end Alembic commands ###
