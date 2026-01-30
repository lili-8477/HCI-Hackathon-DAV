#!/usr/bin/env python3
"""
Enhanced Data Visualization Module with New Features:
- Added pairplot (scatter matrix)
- Added 3D scatter plot
- Added biplot for PCA results
- Added time series plot with decomposition overlays
- Enhanced export options (now supports PDF, SVG)
"""

# ... imports ...

class DataVisualizer:
    """
    Visualization class with new plot types and features.
    """
    
    # Existing methods remain
    
    # NEW: Pairplot (Scatter Matrix)
    def plot_pairplot(self,
                     columns: Optional[List[str]] = None,
                     color_by: Optional[str] = None,
                     title: Optional[str] = None,
                     diag_kind: str = 'hist',
                     filters: Optional[List[Dict]] = None,
                     **kwargs) -> go.Figure:
        """
        Create interactive pairplot (scatter matrix).
        
        Args:
            columns: Columns to include (default: all numeric)
            color_by: Column for coloring
            title: Chart title
            diag_kind: 'hist' or 'kde' for diagonal
            filters: Filters to apply
            
        Returns:
            Plotly Figure
        """
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        if columns is None:
            columns = plot_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("Need at least 2 columns for pairplot")
        
        fig = px.scatter_matrix(
            plot_df,
            dimensions=columns,
            color=color_by,
            color_discrete_sequence=self.get_colors(plot_df[color_by].nunique() if color_by else 1),
            template=self.template,
            **kwargs
        )
        
        # Update diagonal if hist
        if diag_kind == 'hist':
            for i in range(len(columns)):
                fig.update_traces(diagonal_visible=False, showupperhalf=False)
                # Add histograms manually if needed
        
        fig.update_layout(title=title or "Pairplot")
        
        self._log_chart('pairplot', ', '.join(columns), fig.layout.title.text)
        return fig
    
    # NEW: 3D Scatter
    def plot_3d_scatter(self,
                       x: str,
                       y: str,
                       z: str,
                       color_by: Optional[str] = None,
                       size_by: Optional[str] = None,
                       title: Optional[str] = None,
                       filters: Optional[List[Dict]] = None,
                       **kwargs) -> go.Figure:
        """
        Create interactive 3D scatter plot.
        """
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        fig = px.scatter_3d(
            plot_df,
            x=x,
            y=y,
            z=z,
            color=color_by,
            size=size_by,
            color_discrete_sequence=self.get_colors(plot_df[color_by].nunique() if color_by else 1),
            template=self.template,
            **kwargs
        )
        
        fig.update_layout(
            title=title or f"3D Scatter: {x} vs {y} vs {z}",
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            )
        )
        
        self._log_chart('3d_scatter', f"{x},{y},{z}", fig.layout.title.text)
        return fig
    
    # NEW: PCA Biplot
    def plot_pca_biplot(self,
                       pca_result: Dict[str, Any],
                       title: Optional[str] = None,
                       **kwargs) -> go.Figure:
        """
        Create biplot for PCA results.
        
        Args:
            pca_result: Output from compute_pca with biplot=True
            title: Chart title
            
        Returns:
            Plotly Figure with points and loading arrows
        """
        if 'transformed_data' not in pca_result or 'biplot_loadings' not in pca_result:
            raise ValueError("PCA result must include transformed_data and biplot_loadings")
        
        pca_df = pca_result['transformed_data']
        loadings = pd.DataFrame(pca_result['biplot_loadings'])
        
        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter(
            x=pca_df['PC1'],
            y=pca_df['PC2'],
            mode='markers',
            name='Observations',
            marker=dict(color=self.get_colors(1)[0])
        ))
        
        # Add loading arrows
        for feature in loadings.index:
            fig.add_trace(go.Scatter(
                x=[0, loadings.loc[feature, 'PC1']],
                y=[0, loadings.loc[feature, 'PC2']],
                mode='lines+text',
                name=feature,
                text=['', feature],
                textposition='top right',
                line=dict(color='red', width=1, dash='dot')
            ))
        
        fig.update_layout(
            title=title or "PCA Biplot",
            xaxis_title='PC1',
            yaxis_title='PC2',
            template=self.template,
            **kwargs
        )
        
        self._log_chart('pca_biplot', 'PC1 vs PC2 with loadings', fig.layout.title.text)
        return fig
    
    # NEW: Time Series Plot with Decomposition
    def plot_time_series_decomp(self,
                               decomp_result: Dict[str, Any],
                               title: Optional[str] = None,
                               **kwargs) -> go.Figure:
        """
        Plot time series decomposition results.
        
        Args:
            decomp_result: Output from time_series_decomposition
            title: Chart title
            
        Returns:
            Subplot figure with observed, trend, seasonal, residual
        """
        indices = decomp_result['indices']
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
        
        traces = [
            ('observed', 1, 'blue'),
            ('trend', 2, 'green'),
            ('seasonal', 3, 'orange'),
            ('residual', 4, 'red')
        ]
        
        for key, row, color in traces:
            fig.add_trace(go.Scatter(
                x=indices,
                y=decomp_result[key],
                mode='lines',
                name=key.capitalize(),
                line_color=color
            ), row=row, col=1)
        
        fig.update_layout(
            title=title or "Time Series Decomposition",
            height=800,
            template=self.template,
            **kwargs
        )
        
        self._log_chart('ts_decomp', 'Decomposition', fig.layout.title.text)
        return fig
    
    # ENHANCED: Save Figure with more formats
    def save_figure(self,
                   fig: go.Figure,
                   filename: str,
                   format: str = 'html',
                   width: Optional[int] = None,
                   height: Optional[int] = None):
        """
        Save figure with added PDF/SVG support (requires orca/kaleido).
        """
        filepath = Path(filename)
        if format in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            fig.write_image(str(filepath), format=format, width=width, height=height)
        elif format == 'html':
            fig.write_html(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved: {filepath}")

# ... rest of the class ...
