using EnvDTE;
using Microsoft;
using Microsoft.VisualStudio.Shell;
using Microsoft.VisualStudio.Shell.Interop;
using Microsoft.VisualStudio.TextTemplating;
using Microsoft.VisualStudio.TextTemplating.VSHost;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Task = System.Threading.Tasks.Task;

namespace TnelabAutoCode
{
    /// <summary>
    /// This is the class that implements the package exposed by this assembly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The minimum requirement for a class to be considered a valid package for Visual Studio
    /// is to implement the IVsPackage interface and register itself with the shell.
    /// This package uses the helper classes defined inside the Managed Package Framework (MPF)
    /// to do it: it derives from the Package class that provides the implementation of the
    /// IVsPackage interface and uses the registration attributes defined in the framework to
    /// register itself and its components with the shell. These attributes tell the pkgdef creation
    /// utility what data to put into .pkgdef file.
    /// </para>
    /// <para>
    /// To get loaded into VS, the package must be referred by &lt;Asset Type="Microsoft.VisualStudio.VsPackage" ...&gt; in .vsixmanifest file.
    /// </para>
    /// </remarks>
    [ProvideAutoLoad(UIContextGuids.SolutionExists, PackageAutoLoadFlags.BackgroundLoad)]
    [PackageRegistration(UseManagedResourcesOnly = true, AllowsBackgroundLoading = true)]
    [Guid(TnelabAutoCodePackage.PackageGuidString)]
    public sealed class TnelabAutoCodePackage : AsyncPackage
    {
        private DTE dte;
        private Events dte_events;
        private DocumentEvents document_events;
        /// <summary>
        /// TnelabAutoCodePackage GUID string.
        /// </summary>
        public const string PackageGuidString = "33650a75-30ca-4024-8283-9ee9f48fc74f";

        #region Package Members

        /// <summary>
        /// Initialization of the package; this method is called right after the package is sited, so this is the place
        /// where you can put all the initialization code that rely on services provided by VisualStudio.
        /// </summary>
        /// <param name="cancellationToken">A cancellation token to monitor for initialization cancellation, which can occur when VS is shutting down.</param>
        /// <param name="progress">A provider for progress updates.</param>
        /// <returns>A task representing the async work of package initialization, or an already completed task if there is none. Do not return null from this method.</returns>
        protected override async Task InitializeAsync(CancellationToken cancellationToken, IProgress<ServiceProgressData> progress)
        {
            // When initialized asynchronously, the current thread may be a background thread at this point.
            // Do any initialization that requires the UI thread after switching to the UI thread.
            await this.JoinableTaskFactory.SwitchToMainThreadAsync(cancellationToken);
            dte = await GetServiceAsync(typeof(SDTE)) as DTE;
            Assumes.Present(dte);
            dte_events = dte.Events;
            document_events = dte_events.DocumentEvents;
            document_events.DocumentSaved += (doc) => {
                ThreadHelper.ThrowIfNotOnUIThread();
                var tdoc = doc.Object("TextDocument") as TextDocument;
                var sp = tdoc.StartPoint;
                var ep = sp.CreateEditPoint();
                //ep.StartOfDocument();
                int num1 = 1, num2 = 2;
                while (true)
                {
                    var txt = ep.GetLines(num1, num2).Trim();
                    if (txt.IndexOf("tne://exec_t4(") != -1)
                    {
                        ExecT4(doc, txt);
                    }
                    else
                    {
                        break;
                    }
                    num1++;
                    num2++;
                }
            };
        }
        void ExecT4(Document doc, string tneTxt)
        {
            try
            {
                ThreadHelper.ThrowIfNotOnUIThread();
                var zuoKuaHaoIndex = tneTxt.IndexOf('(');
                var youKuaHaoIndex = tneTxt.IndexOf(')');
                var t4Infos = tneTxt.Substring(zuoKuaHaoIndex + 1, youKuaHaoIndex - zuoKuaHaoIndex - 1).Split('?');
                var t4Path = t4Infos[0];
                var currentPath = Path.GetDirectoryName(doc.FullName);
                if (t4Path.IndexOf(":") == -1)
                {
                    t4Path = Path.Combine(currentPath, t4Path);
                }
                if (File.Exists(t4Path))
                {
                    var t4 = GetService(typeof(STextTemplating)) as ITextTemplating;
                    if (t4 == null)
                        return;
                    if (t4Infos.Length == 2)
                    {
                        var sessionT4 = t4 as ITextTemplatingSessionHost;
                        if (sessionT4 != null)
                        {
                            sessionT4.Session = sessionT4.CreateSession();
                            var paramInfos = t4Infos[1].Split('&');
                            for (var i = 0; i < paramInfos.Length; i++)
                            {
                                var valInfos = paramInfos[i].Split('=');
                                sessionT4.Session[valInfos[0]] = valInfos[1];
                            }
                        }
                    }
                    var t4Text = File.ReadAllText(t4Path);
                    string result = t4.ProcessTemplate(t4Path, t4Text, null);
                }
            }
            catch
            {
            }
        }
        #endregion
    }
}
